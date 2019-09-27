import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Dense, Flatten, concatenate

activation_fn = tf.keras.activations.tanh

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0.0, .1),
    'bias_initializer': tf.constant_initializer(0.1)    # 1.x 需要指定dtype
}

class ImageNet(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)
        self.conv1 = Conv3D(filters=32, kernel_size=[1, 8, 8], strides=[1, 4, 4], padding='valid', activation=activation_fn, name='conv1', **initKernelAndBias)
        self.conv2 = Conv3D(filters=64, kernel_size=[1, 4, 4], strides=[1, 2, 2], padding='valid', activation=activation_fn, name='conv2', **initKernelAndBias)
        self.conv3 = Conv3D(filters=64, kernel_size=[1, 3, 3], strides=[1, 1, 1], padding='valid', activation=activation_fn, name='conv3', **initKernelAndBias)
        self.flatten1 = Flatten()
        self.fc1 = Dense(256, activation_fn, name='fc1', **initKernelAndBias)

    def call(self, vector_input, visual_input):
        if len(visual_input.shape) != 5:
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
    def __init__(self, output_shape, name):
        super().__init__(name=name)
        self.layer1 = Dense(128, activation_fn, name='actor1', **initKernelAndBias)
        self.layer2 = Dense(64, activation_fn, name='actor2', **initKernelAndBias)
        self.action_probs = Dense(output_shape, tf.keras.activations.softmax, name='action_probs', **initKernelAndBias)

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
    def __init__(self, output_shape, name):
        super().__init__(name=name)
        self.layer1 = Dense(128, activation_fn, name='actor1', **initKernelAndBias)
        self.layer2 = Dense(64, activation_fn, name='actor2', **initKernelAndBias)
        self.mu = Dense(output_shape, tf.keras.activations.tanh, name='mu', **initKernelAndBias)
        self.sigma1 = Dense(64, activation_fn, name='sigma1', **initKernelAndBias)
        self.sigma = Dense(output_shape, tf.keras.activations.sigmoid, name='sigma', **initKernelAndBias)

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
    def __init__(self, output_shape, name):
        super().__init__(name=name)
        self.layer1 = Dense(128, activation_fn, name='actor1', **initKernelAndBias)
        self.layer2 = Dense(64, activation_fn, name='actor2', **initKernelAndBias)
        self.mu = Dense(output_shape, tf.keras.activations.tanh, name='mu', **initKernelAndBias)

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
    def __init__(self, name):
        super().__init__(name=name)
        self.layer1 = Dense(256, activation_fn, name='layer1', **initKernelAndBias)
        self.layer2 = Dense(256, activation_fn, name='layer2', **initKernelAndBias)
        self.q = Dense(1, None, name='value', **initKernelAndBias)

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
    def __init__(self, name):
        super().__init__(name=name)
        self.critic1 = Dense(256, activation_fn, name='critic1', **initKernelAndBias)
        self.critic2 = Dense(256, activation_fn, name='critic2', **initKernelAndBias)
        self.v = Dense(1, None, name='v', **initKernelAndBias)

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
    def __init__(self, output_shape, name):
        super().__init__(name=name)
        self.layer1 = Dense(256, activation_fn, name='layer1', **initKernelAndBias)
        self.layer2 = Dense(256, activation_fn, name='layer2', **initKernelAndBias)
        self.q = Dense(output_shape, None, name='value', **initKernelAndBias)

    def call(self, vector_input, visual_input):
        features = self.layer1(super().call(vector_input, visual_input))
        features = self.layer2(features)
        q = self.q(features)
        return q

class critic_dueling(ImageNet):
    def __init__(self, output_shape, name):
        super().__init__(name=name)
        self.layer1 = Dense(256, activation_fn, name='layer1', **initKernelAndBias)
        self.layer2 = Dense(256, activation_fn, name='layer2', **initKernelAndBias)
        self.layer3 = Dense(256, activation_fn, name='layer3', **initKernelAndBias)
        self.v = Dense(1, None, name='value', **initKernelAndBias)
        self.a = Dense(output_shape, None, name='advantage', **initKernelAndBias)

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
    def __init__(self, output_shape, name):
        super().__init__(name=name)
        self.share1 = Dense(512, activation_fn, name='share1', **initKernelAndBias)
        self.share2 = Dense(256, activation_fn, name='share2', **initKernelAndBias)
        self.actor1 = Dense(128, activation_fn, name='actor1', **initKernelAndBias)
        self.actor2 = Dense(64, activation_fn, name='actor2', **initKernelAndBias)
        self.action_probs = Dense(output_shape, tf.keras.activations.softmax, name='action_probs', **initKernelAndBias)
        self.critic1 = Dense(128, activation_fn, name='critic1', **initKernelAndBias)
        self.critic2 = Dense(64, activation_fn, name='critic2', **initKernelAndBias)
        self.value = Dense(1, None, name='value', **initKernelAndBias)

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
    def __init__(self, output_shape, name):
        super().__init__(name=name)
        self.share1 = Dense(512, activation_fn, name='share1', **initKernelAndBias)
        self.share2 = Dense(256, activation_fn, name='share2', **initKernelAndBias)
        self.actor1 = Dense(128, activation_fn, name='actor1', **initKernelAndBias)
        self.actor2 = Dense(64, activation_fn, name='actor2', **initKernelAndBias)
        self.mu = Dense(output_shape, tf.keras.activations.tanh, name='action_probs', **initKernelAndBias)
        self.sigma1 = Dense(64, activation_fn, name='sigma1', **initKernelAndBias)
        self.sigma = Dense(output_shape, tf.keras.activations.sigmoid, name='sigma', **initKernelAndBias)
        self.critic1 = Dense(128, activation_fn, name='critic1', **initKernelAndBias)
        self.critic2 = Dense(64, activation_fn, name='critic2', **initKernelAndBias)
        self.value = Dense(1, None, name='value', **initKernelAndBias)

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
