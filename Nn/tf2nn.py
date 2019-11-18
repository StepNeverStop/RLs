import tensorflow as tf
from .activations import swish, mish
from tensorflow.keras import Sequential
from tensorflow.python.framework import tensor_shape
from tensorflow.keras.layers import Conv2D, Conv3D, Dense, Flatten, GaussianNoise

activation_fn = tf.keras.activations.tanh

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0.0, .1),
    'bias_initializer': tf.constant_initializer(0.1)    # 2.x 不需要指定dtype
}


class mlp(Sequential):
    def __init__(self, hidden_units, act_fn=activation_fn, output_shape=1, out_activation=None, out_layer=True):
        """
        inputs:
            hidden_units: like [32, 32]
            output_shape: units of last layer
            out_activation: activation function of last layer
            out_layer: whether need specifing last layer or not
        """
        super().__init__()
        for u in hidden_units:
            self.add(Dense(u, act_fn))
        if out_layer:
            self.add(Dense(output_shape, out_activation))


class mlp_witch_noisy(Sequential):
    def __init__(self, hidden_units, act_fn=activation_fn, output_shape=1, out_activation=None, out_layer=True):
        """
        inputs:
            hidden_units: like [32, 32]
            output_shape: units of last layer
            out_activation: activation function of last layer
            out_layer: whether need specifing last layer or not
        """
        super().__init__()
        for u in hidden_units:
            self.add(GaussianNoise(0.4))        #Or use kwargs
            self.add(Dense(u, act_fn))
        if out_layer:
            self.add(GaussianNoise(0.4))
            self.add(Dense(output_shape, out_activation))


class Noisy(Dense):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(units, activation=None, **kwargs)
        self.noise_sigma = 0.4 if 'noise_sigma' not in kwargs.keys() else kwargs['noise_sigma']

    def build(self, input_shape):
        super().build(input_shape)
        self.build = False
        self.last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.noisy_w = self.add_weight(
            'noise_kernel',
            shape=[self.last_dim, self.units],
            initializer=tf.random_normal_initializer(0.0, .1),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.noisy_b = self.add_weight(
                'noise_bias',
                shape=[self.units, ],
                initializer=tf.constant_initializer(self.noise_sigma / np.sqrt(self.units)),
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.build = True

    def noisy_layer(self, inputs):
        epsilon_w = tf.random.truncated_normal([self.last_dim, self.units], stddev=self.noise_sigma)
        epsilon_b = tf.random.truncated_normal([self.units, ], stddev=self.noise_sigma)
        return tf.matmul(inputs, self.noisy_w * epsilon_w) + self.noisy_b * epsilon_b

    def call(self, inputs):
        y = super().call(inputs)
        noise = self.noisy_layer(inputs)
        return y + noise


class ImageNet(tf.keras.Model):
    def __init__(self, name, visual_dim=[]):
        super().__init__(name=name)
        self.build_visual = False
        if len(visual_dim) == 4:
            self.conv1 = Conv3D(filters=32, kernel_size=[1, 8, 8], strides=[1, 4, 4], padding='valid', activation=activation_fn)
            self.conv2 = Conv3D(filters=64, kernel_size=[1, 4, 4], strides=[1, 2, 2], padding='valid', activation=activation_fn)
            self.conv3 = Conv3D(filters=64, kernel_size=[1, 3, 3], strides=[1, 1, 1], padding='valid', activation=activation_fn)
            self.flatten = Flatten()
            self.fc = Dense(128, activation_fn)
            self.build_visual = True
        elif len(visual_dim) == 3:
            self.conv1 = Conv2D(filters=32, kernel_size=[8, 8], strides=[4, 4], padding='valid', activation=activation_fn)
            self.conv2 = Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], padding='valid', activation=activation_fn)
            self.conv3 = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='valid', activation=activation_fn)
            self.flatten = Flatten()
            self.fc = Dense(128, activation_fn)
            self.build_visual = True

    def call(self, vector_input, visual_input):
        if self.build_visual:
            features = self.conv1(visual_input)
            features = self.conv2(features)
            features = self.conv3(features)
            features = self.flatten(features)
            features = self.fc(features)
            vector_input = tf.concat((features, vector_input), axis=-1)
        return vector_input


class actor_dpg(ImageNet):
    '''
    use for DDPG and/or TD3 algorithms' actor network.
    input: vector of state
    output: deterministic action(mu) and disturbed action(action) given a state
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name, hidden_units):
        super().__init__(name=name, visual_dim=visual_dim)
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation=tf.keras.activations.tanh)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        mu = self.net(super().call(vector_input, visual_input))
        return mu


class actor_mu(ImageNet):
    '''
    use for PPO/PG algorithms' actor network.
    input: vector of state
    output: stochastic action(mu), normally is the mean of a Normal distribution
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name, hidden_units):
        super().__init__(name=name, visual_dim=visual_dim)
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation='tanh')
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        mu = self.net(super().call(vector_input, visual_input))
        return mu


class actor_continuous(ImageNet):
    '''
    use for continuous action space.
    input: vector of state
    output: mean(mu) and log_variance(log_std) of Gaussian Distribution of actions given a state
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name, hidden_units):
        super().__init__(name=name, visual_dim=visual_dim)
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.mu = mlp(hidden_units['mu'], output_shape=output_shape, out_activation=None)
        self.log_std = mlp(hidden_units['log_std'], output_shape=output_shape, out_activation=tf.keras.activations.tanh)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.share(super().call(vector_input, visual_input))
        mu = self.mu(features)
        log_std = self.log_std(features)
        return mu, log_std


class actor_discrete(ImageNet):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name, hidden_units):
        super().__init__(name=name, visual_dim=visual_dim)
        self.logits = mlp(hidden_units, output_shape=output_shape, out_activation=None)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        logits = self.logits(super().call(vector_input, visual_input))
        return logits


class critic_q_one(ImageNet):
    '''
    use for evaluate the value given a state-action pair.
    input: tf.concat((state, action),axis = 1)
    output: q(s,a)
    '''

    def __init__(self, vector_dim, visual_dim, action_dim, name, hidden_units):
        super().__init__(name=name, visual_dim=visual_dim)
        self.net = mlp(hidden_units, output_shape=1, out_activation=None)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim), tf.keras.Input(shape=action_dim))

    def call(self, vector_input, visual_input, action):
        features = tf.concat((super().call(vector_input, visual_input), action), axis=-1)
        q = self.net(features)
        return q


class critic_v(ImageNet):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''

    def __init__(self, vector_dim, visual_dim, name, hidden_units):
        super().__init__(name=name, visual_dim=visual_dim)
        self.net = mlp(hidden_units, output_shape=1, out_activation=None)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        v = self.net(super().call(vector_input, visual_input))
        return v


class critic_q_all(ImageNet):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name, hidden_units):
        super().__init__(name=name, visual_dim=visual_dim)
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation=None)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        q = self.net(super().call(vector_input, visual_input))
        return q


class critic_dueling(ImageNet):
    def __init__(self, vector_dim, visual_dim, output_shape, name, hidden_units):
        super().__init__(name=name, visual_dim=visual_dim)
        self.share = mlp_witch_noisy(hidden_units['share'], out_layer=False)
        self.v = mlp_witch_noisy(hidden_units['v'], output_shape=1, out_activation=None)
        self.adv = mlp_witch_noisy(hidden_units['adv'], output_shape=output_shape, out_activation=None)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.share(super().call(vector_input, visual_input))
        v = self.v(features)
        adv = self.adv(features)
        return v, adv


class a_c_v_continuous(ImageNet):
    '''
    combine actor network and critic network, share some nn layers. use for continuous action space.
    input: vector of state
    output: mean(mu) of Gaussian Distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name, hidden_units):
        super().__init__(name=name, visual_dim=visual_dim)
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.mu = mlp(hidden_units['mu'], output_shape=output_shape, out_activation='tanh')
        self.v = mlp(hidden_units['v'], output_shape=1, out_activation=None)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.share(super().call(vector_input, visual_input))
        v = self.v(features)
        mu = self.mu(features)
        return mu, v


class a_c_v_discrete(ImageNet):
    '''
    combine actor network and critic network, share some nn layers. use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name, hidden_units):
        super().__init__(name=name, visual_dim=visual_dim)
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.logits = mlp(hidden_units['logits'], output_shape=output_shape, out_activation=None)
        self.v = mlp(hidden_units['v'], output_shape=1, out_activation=None)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.share(super().call(vector_input, visual_input))
        logits = self.logits(features)
        v = self.v(features)
        return logits, v
