import tensorflow as tf
from .activations import swish, mish
from tensorflow.keras.layers import Conv3D, Dense, Flatten, concatenate

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
        self.flatten1 = Flatten()
        self.fc1 = Dense(256, activation_fn, **initKernelAndBias)

    def call(self, vector_input, visual_input):
        if visual_input is None or len(visual_input.shape) != 5:
            pass
        else:
            features = self.conv1(visual_input)
            features = self.conv2(features)
            features = self.conv3(features)
            features = self.flatten1(features)
            features = self.fc1(features)
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
        self.layer1 = Dense(128, activation_fn, **initKernelAndBias)
        self.layer2 = Dense(64, activation_fn, **initKernelAndBias)
        self.action_probs = Dense(output_shape, tf.keras.activations.softmax, **initKernelAndBias)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.layer1(super().call(vector_input, visual_input))
        features = self.layer2(features)
        action_probs = self.action_probs(features)
        return action_probs


class actor_continuous(ImageNet):
    '''
    use for continuous action space.
    input: vector of state
    output: mean(mu) and variance(sigma) of Gaussian Distribution of actions given a state
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name):
        super().__init__(name=name)
        self.layer1 = Dense(128, activation_fn, **initKernelAndBias)
        self.layer2 = Dense(64, activation_fn, **initKernelAndBias)
        self.mu = Dense(output_shape, tf.keras.activations.tanh, **initKernelAndBias)
        self.sigma1 = Dense(64, activation_fn, **initKernelAndBias)
        self.sigma = Dense(output_shape, tf.keras.activations.sigmoid, **initKernelAndBias)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.layer1(super().call(vector_input, visual_input))
        features = self.layer2(features)
        mu = self.mu(features)
        features = self.sigma1(features)
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
        self.layer1 = Dense(128, activation_fn, **initKernelAndBias)
        self.layer2 = Dense(64, activation_fn, **initKernelAndBias)
        self.mu = Dense(output_shape, tf.keras.activations.tanh, **initKernelAndBias)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.layer1(super().call(vector_input, visual_input))
        features = self.layer2(features)
        mu = self.mu(features)
        return mu


class critic_q_one(ImageNet):
    '''
    use for evaluate the value given a state-action pair.
    input: tf.concat((state, action),axis = 1)
    output: q(s,a)
    '''

    def __init__(self, vector_dim, visual_dim, name):
        super().__init__(name=name)
        self.layer1 = Dense(256, activation_fn, **initKernelAndBias)
        self.layer2 = Dense(256, activation_fn, **initKernelAndBias)
        self.q = Dense(1, None, **initKernelAndBias)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input, action):
        features = tf.concat((super().call(vector_input, visual_input), action), axis=-1)
        features = self.layer1(features)
        features = self.layer2(features)
        q = self.q(features)
        return q


class critic_v(ImageNet):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''

    def __init__(self, vector_dim, visual_dim, name):
        super().__init__(name=name)
        self.critic1 = Dense(256, activation_fn, **initKernelAndBias)
        self.critic2 = Dense(256, activation_fn, **initKernelAndBias)
        self.v = Dense(1, None, **initKernelAndBias)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.critic1(super().call(vector_input, visual_input))
        features = self.critic2(features)
        v = self.v(features)
        return v


class critic_q_all(ImageNet):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name):
        super().__init__(name=name)
        self.layer1 = Dense(256, activation_fn, **initKernelAndBias)
        self.layer2 = Dense(256, activation_fn, **initKernelAndBias)
        self.q = Dense(output_shape, None, **initKernelAndBias)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.layer1(super().call(vector_input, visual_input))
        features = self.layer2(features)
        q = self.q(features)
        return q


class critic_dueling(ImageNet):
    def __init__(self, vector_dim, visual_dim, output_shape, name):
        super().__init__(name=name)
        self.layer1 = Dense(256, activation_fn, **initKernelAndBias)
        self.layer2 = Dense(256, activation_fn, **initKernelAndBias)
        self.layer3 = Dense(256, activation_fn, **initKernelAndBias)
        self.v = Dense(1, None, **initKernelAndBias)
        self.a = Dense(output_shape, None, **initKernelAndBias)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.layer1(super().call(vector_input, visual_input))
        features1 = self.layer2(features)
        features2 = self.layer3(features)
        v = self.v(features1)
        a = self.a(features2)
        return v, a


class a_c_v_discrete(ImageNet):
    '''
    combine actor network and critic network, share some nn layers. use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name):
        super().__init__(name=name)
        self.share1 = Dense(512, activation_fn, **initKernelAndBias)
        self.share2 = Dense(256, activation_fn, **initKernelAndBias)
        self.actor1 = Dense(128, activation_fn, **initKernelAndBias)
        self.actor2 = Dense(64, activation_fn, **initKernelAndBias)
        self.action_probs = Dense(output_shape, tf.keras.activations.softmax, **initKernelAndBias)
        self.critic1 = Dense(128, activation_fn, **initKernelAndBias)
        self.critic2 = Dense(64, activation_fn, **initKernelAndBias)
        self.value = Dense(1, None, **initKernelAndBias)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.share1(super().call(vector_input, visual_input))
        features = self.share2(features)
        features1 = self.actor1(features)
        features1 = self.actor2(features1)
        action_probs = self.action_probs(features1)
        features2 = self.critic1(features)
        features2 = self.critic2(features2)
        value = self.value(features2)
        return action_probs, value


class a_c_v_continuous(ImageNet):
    '''
    combine actor network and critic network, share some nn layers. use for continuous action space.
    input: vector of state
    output: mean(mu) and variance(sigma) of Gaussian Distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, visual_dim, output_shape, name):
        super().__init__(name=name)
        self.share1 = Dense(512, activation_fn, **initKernelAndBias)
        self.share2 = Dense(256, activation_fn, **initKernelAndBias)
        self.actor1 = Dense(128, activation_fn, **initKernelAndBias)
        self.actor2 = Dense(64, activation_fn, **initKernelAndBias)
        self.mu = Dense(output_shape, tf.keras.activations.tanh, **initKernelAndBias)
        self.sigma1 = Dense(64, activation_fn, **initKernelAndBias)
        self.sigma = Dense(output_shape, tf.keras.activations.sigmoid, **initKernelAndBias)
        self.critic1 = Dense(128, activation_fn, **initKernelAndBias)
        self.critic2 = Dense(64, activation_fn, **initKernelAndBias)
        self.value = Dense(1, None, **initKernelAndBias)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=visual_dim))

    def call(self, vector_input, visual_input):
        features = self.share1(super().call(vector_input, visual_input))
        features = self.share2(features)
        features1 = self.actor1(features)
        features1_1 = self.actor2(features1)
        mu = self.mu(features1_1)
        features1_2 = self.sigma1(features1)
        sigma = self.sigma(features1_2)
        features2 = self.critic1(features)
        features2 = self.critic2(features2)
        value = self.value(features2)
        return mu, sigma, value
