import tensorflow as tf
from .activations import swish, mish

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0.0, .1),
    'bias_initializer': tf.constant_initializer(0.1, dtype=tf.float32)
}

activation_fn = tf.nn.tanh

def get_state(name, vector_input, visual_input):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if visual_input == None or len(visual_input.shape) == 2:
            return vector_input
        else:
            conv1 = tf.layers.conv3d(inputs=visual_input, filters=32, kernel_size=[1, 8, 8], strides=[1, 4, 4], padding='valid', activation=activation_fn, name='conv1', **initKernelAndBias)
            conv2 = tf.layers.conv3d(inputs=conv1, filters=64, kernel_size=[1, 4, 4], strides=[1, 2, 2], padding='valid', activation=activation_fn, name='conv2', **initKernelAndBias)
            conv3 = tf.layers.conv3d(inputs=conv2, filters=64, kernel_size=[1, 3, 3], strides=[1, 1, 1], padding='valid', activation=activation_fn, name='conv3', **initKernelAndBias)
            feature = tf.layers.dense(tf.layers.flatten(conv3), 256, activation_fn, name='fc1', **initKernelAndBias)
            state = tf.concat((feature, vector_input), axis=-1)
            return state


def actor_discrete(name, vector_input, visual_input, out_shape):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''
    state = get_state(name, vector_input, visual_input)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        actor1 = tf.layers.dense(state, 128, activation_fn, name='actor1', **initKernelAndBias)
        actor2 = tf.layers.dense(actor1, 64, activation_fn, name='actor2', **initKernelAndBias)
        action_probs = tf.layers.dense(actor2, out_shape, tf.nn.softmax, name='action_probs', **initKernelAndBias)
        return action_probs


def actor_continuous(name, vector_input, visual_input, out_shape):
    '''
    use for continuous action space.
    input: vector of state
    output: mean(mu) and variance(sigma) of Gaussian Distribution of actions given a state
    '''
    state = get_state(name, vector_input, visual_input)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        actor1 = tf.layers.dense(state, 128, activation_fn, name='actor1', **initKernelAndBias)
        actor2 = tf.layers.dense(actor1, 64, activation_fn, name='actor2', **initKernelAndBias)
        mu = tf.layers.dense(actor2, out_shape, tf.nn.tanh, name='mu', **initKernelAndBias)
        sigma1 = tf.layers.dense(actor1, 64, activation_fn, name='sigma1', **initKernelAndBias)
        sigma = tf.layers.dense(sigma1, out_shape, tf.nn.sigmoid, name='sigma', **initKernelAndBias)
        return mu, sigma


def actor_dpg(name, vector_input, visual_input, out_shape):
    '''
    use for DDPG and/or TD3 algorithms' actor network.
    input: vector of state
    output: deterministic action(mu) and disturbed action(action) given a state
    '''
    state = get_state(name, vector_input, visual_input)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        actor1 = tf.layers.dense(state, 128, activation_fn, name='actor1', **initKernelAndBias)
        actor2 = tf.layers.dense(actor1, 64, activation_fn, name='actor2', **initKernelAndBias)
        mu = tf.layers.dense(actor2, out_shape, tf.nn.tanh, name='mu', **initKernelAndBias)
        return mu


def critic_q_one(name, vector_input, visual_input, action):
    '''
    use for evaluate the value given a state-action pair.
    input: tf.concat((state, action),axis = 1)
    output: q(s,a)
    '''
    state = get_state(name, vector_input, visual_input)
    s_a = tf.concat((state, action), axis=-1)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        critic1 = tf.layers.dense(s_a, 256, activation_fn, name='critic1', **initKernelAndBias)
        critic2 = tf.layers.dense(critic1, 256, activation_fn, name='critic2', **initKernelAndBias)
        q = tf.layers.dense(critic2, 1, None, name='q', **initKernelAndBias)
        # var = tf.get_variable_scope().global_variables()
        return q


def critic_v(name, vector_input, visual_input):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''
    state = get_state(name, vector_input, visual_input)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        critic1 = tf.layers.dense(state, 256, activation_fn, name='critic1', **initKernelAndBias)
        critic2 = tf.layers.dense(critic1, 256, activation_fn, name='critic2', **initKernelAndBias)
        v = tf.layers.dense(critic2, 1, None, name='v', **initKernelAndBias)
        return v


def critic_q_all(name, vector_input, visual_input, out_shape):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''
    state = get_state(name, vector_input, visual_input)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        layer1 = tf.layers.dense(state, 256, activation_fn, name='layer1', **initKernelAndBias)
        layer2 = tf.layers.dense(layer1, 256, activation_fn, name='layer2', **initKernelAndBias)
        q = tf.layers.dense(layer2, out_shape, None, name='value', **initKernelAndBias)
        return q

def critic_dueling(name, vector_input, visual_input, out_shape):
    state = get_state(name, vector_input, visual_input)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        layer1 = tf.layers.dense(state, 256, activation_fn, name='layer1', **initKernelAndBias)
        layer2 = tf.layers.dense(layer1, 256, activation_fn, name='layer2', **initKernelAndBias)
        layer3 = tf.layers.dense(layer1, 256, activation_fn, name='layer3', **initKernelAndBias)
        v = tf.layers.dense(layer2, 1, None, name='value', **initKernelAndBias)
        a = tf.layers.dense(layer3, out_shape, None, name='advantage', **initKernelAndBias)
        return v, a


def a_c_v_discrete(name, vector_input, visual_input, out_shape):
    '''
    combine actor network and critic network, share some nn layers. use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state, v(s)
    '''
    state = get_state(name, vector_input, visual_input)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        share1 = tf.layers.dense(state, 512, activation_fn, name='share1', **initKernelAndBias)
        share2 = tf.layers.dense(share1, 256, activation_fn, name='share2', **initKernelAndBias)
        actor1 = tf.layers.dense(share2, 128, activation_fn, name='actor1', **initKernelAndBias)
        actor2 = tf.layers.dense(actor1, 64, activation_fn, name='actor2', **initKernelAndBias)
        action_probs = tf.layers.dense(actor2, out_shape, tf.nn.softmax, name='action_probs', **initKernelAndBias)
        critic1 = tf.layers.dense(share2, 128, activation_fn, name='critic1', **initKernelAndBias)
        critic2 = tf.layers.dense(critic1, 64, activation_fn, name='critic2', **initKernelAndBias)
        value = tf.layers.dense(critic2, 1, None, name='value', **initKernelAndBias)
        return action_probs, value


def a_c_v_continuous(name, vector_input, visual_input, out_shape):
    '''
    combine actor network and critic network, share some nn layers. use for continuous action space.
    input: vector of state
    output: mean(mu) and variance(sigma) of Gaussian Distribution of actions given a state, v(s)
    '''
    state = get_state(name, vector_input, visual_input)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        share1 = tf.layers.dense(state, 512, activation_fn, name='share1', **initKernelAndBias)
        share2 = tf.layers.dense(share1, 256, activation_fn, name='share2', **initKernelAndBias)
        actor1 = tf.layers.dense(share2, 128, activation_fn, name='actor1', **initKernelAndBias)
        actor2 = tf.layers.dense(actor1, 64, activation_fn, name='actor2', **initKernelAndBias)
        mu = tf.layers.dense(actor2, out_shape, tf.nn.tanh, name='mu', **initKernelAndBias)
        sigma1 = tf.layers.dense(actor1, 64, activation_fn, name='sigma1', **initKernelAndBias)
        sigma = tf.layers.dense(sigma1, out_shape, tf.nn.sigmoid, name='sigma', **initKernelAndBias)
        critic1 = tf.layers.dense(share2, 128, activation_fn, name='critic1', **initKernelAndBias)
        critic2 = tf.layers.dense(critic1, 64, activation_fn, name='critic2', **initKernelAndBias)
        value = tf.layers.dense(critic2, 1, None, name='value', **initKernelAndBias)
        return mu, sigma, value
