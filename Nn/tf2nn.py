import tensorflow as tf
from .activations import swish, mish
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv3D, Dense, Flatten

activation_fn = tf.keras.activations.tanh

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0.0, .1),
    'bias_initializer': tf.constant_initializer(0.1)    # 2.x 不需要指定dtype
}


class ImageNet(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)
        self.conv1 = Conv3D(filters=32, kernel_size=[1, 8, 8], strides=[1, 4, 4], padding='valid', activation=activation_fn, **initKernelAndBias)
        self.conv2 = Conv3D(filters=64, kernel_size=[1, 4, 4], strides=[1, 2, 2], padding='valid', activation=activation_fn, **initKernelAndBias)
        self.conv3 = Conv3D(filters=64, kernel_size=[1, 3, 3], strides=[1, 1, 1], padding='valid', activation=activation_fn, **initKernelAndBias)
        self.flatten = Flatten()
        self.fc = Dense(256, activation_fn, **initKernelAndBias)

    def call(self, vector_input, visual_input):
        if visual_input is None or len(visual_input.shape) != 5:
            pass
        else:
            features = self.conv1(visual_input)
            features = self.conv2(features)
            features = self.conv3(features)
            features = self.flatten(features)
            features = self.fc(features)
            vector_input = tf.concat((features, vector_input), axis=-1)
        return vector_input


class actor_discrete(ImageNet):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name):
        super().__init__(name=name)
        self.net = Sequential([
            Dense(128, activation_fn, **initKernelAndBias),
            Dense(64, activation_fn, **initKernelAndBias),
            Dense(output_shape, tf.keras.activations.softmax, **initKernelAndBias)
        ])
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        action_probs = self.net(super().call(vector_input, visual_input))
        return action_probs


class actor_continuous(ImageNet):
    '''
    use for continuous action space.
    input: vector of state
    output: mean(mu) and variance(sigma) of Gaussian Distribution of actions given a state
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name):
        super().__init__(name=name)
        self.share = Sequential([
            Dense(128, activation_fn, **initKernelAndBias),
            Dense(64, activation_fn, **initKernelAndBias)
        ])
        self.mu = Dense(output_shape, tf.keras.activations.tanh, **initKernelAndBias)
        self.sigma = Sequential([
            Dense(64, activation_fn, **initKernelAndBias),
            Dense(output_shape, tf.keras.activations.sigmoid, **initKernelAndBias)
        ])
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.share(super().call(vector_input, visual_input))
        mu = self.mu(features)
        sigma = self.sigma(features)
        return mu, sigma


class actor_dpg(ImageNet):
    '''
    use for DDPG and/or TD3 algorithms' actor network.
    input: vector of state
    output: deterministic action(mu) and disturbed action(action) given a state
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name):
        super().__init__(name=name)
        self.net = Sequential([
            Dense(128, activation_fn, **initKernelAndBias),
            Dense(64, activation_fn, **initKernelAndBias),
            Dense(output_shape, tf.keras.activations.tanh, **initKernelAndBias)
        ])
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        mu = self.net(super().call(vector_input, visual_input))
        return mu


class critic_q_one(ImageNet):
    '''
    use for evaluate the value given a state-action pair.
    input: tf.concat((state, action),axis = 1)
    output: q(s,a)
    '''

    def __init__(self, vector_dim, visual_dim, action_dim, name):
        super().__init__(name=name)
        self.net = Sequential([
            Dense(256, activation_fn, **initKernelAndBias),
            Dense(256, activation_fn, **initKernelAndBias),
            Dense(1, None, **initKernelAndBias)
        ])
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

    def __init__(self, vector_dim, visual_dim, name):
        super().__init__(name=name)
        self.net = Sequential([
            Dense(256, activation_fn, **initKernelAndBias),
            Dense(256, activation_fn, **initKernelAndBias),
            Dense(1, None, **initKernelAndBias)
        ])
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

    def __init__(self, vector_dim, visual_dim, output_shape, name):
        super().__init__(name=name)
        self.net = Sequential([
            Dense(256, activation_fn, **initKernelAndBias),
            Dense(256, activation_fn, **initKernelAndBias),
            Dense(output_shape, None, **initKernelAndBias)
        ])
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        q = self.net(super().call(vector_input, visual_input))
        return q


class critic_dueling(ImageNet):
    def __init__(self, vector_dim, visual_dim, output_shape, name):
        super().__init__(name=name)
        self.share = Dense(256, activation_fn, **initKernelAndBias)
        self.v = Sequential([
            Dense(256, activation_fn, **initKernelAndBias),
            Dense(1, None, **initKernelAndBias)
        ])
        self.a = Sequential([
            Dense(256, activation_fn, **initKernelAndBias),
            Dense(output_shape, None, **initKernelAndBias)
        ])
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.share(super().call(vector_input, visual_input))
        v = self.v(features)
        a = self.a(features)
        return v, a


class a_c_v_discrete(ImageNet):
    '''
    combine actor network and critic network, share some nn layers. use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name):
        super().__init__(name=name)
        self.share = Sequential([
            Dense(512, activation_fn, **initKernelAndBias),
            Dense(256, activation_fn, **initKernelAndBias)
        ])
        self.action_probs = Sequential([
            Dense(128, activation_fn, **initKernelAndBias),
            Dense(64, activation_fn, **initKernelAndBias),
            Dense(output_shape, tf.keras.activations.softmax, **initKernelAndBias)
        ])
        self.value = Sequential([
            Dense(128, activation_fn, **initKernelAndBias),
            Dense(64, activation_fn, **initKernelAndBias),
            Dense(1, None, **initKernelAndBias)
        ])
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.share(super().call(vector_input, visual_input))
        action_probs = self.action_probs(features)
        value = self.value(features)
        return action_probs, value


class a_c_v_continuous(ImageNet):
    '''
    combine actor network and critic network, share some nn layers. use for continuous action space.
    input: vector of state
    output: mean(mu) and variance(sigma) of Gaussian Distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name):
        super().__init__(name=name)
        self.share = Sequential([
            Dense(512, activation_fn, **initKernelAndBias),
            Dense(256, activation_fn, **initKernelAndBias)
        ])
        self.actor = Dense(128, activation_fn, **initKernelAndBias)
        self.mu = Sequential([
            Dense(64, activation_fn, **initKernelAndBias),
            Dense(output_shape, tf.keras.activations.tanh, **initKernelAndBias)
        ])
        self.sigma = Sequential([
            Dense(64, activation_fn, **initKernelAndBias),
            Dense(output_shape, tf.keras.activations.sigmoid, **initKernelAndBias)
        ])
        self.value = Sequential([
            Dense(128, activation_fn, **initKernelAndBias),
            Dense(64, activation_fn, **initKernelAndBias),
            Dense(1, None, **initKernelAndBias)
        ])
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.share(super().call(vector_input, visual_input))
        value = self.value(features)
        features_actor = self.actor(features)
        mu = self.mu(features_actor)
        sigma = self.sigma(features_actor)
        return mu, sigma, value
