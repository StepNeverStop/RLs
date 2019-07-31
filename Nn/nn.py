import tensorflow as tf

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0.0, .1),
    'bias_initializer': tf.constant_initializer(0.1, dtype=tf.float32)
}

activation_fn = tf.nn.tanh


def visual_nn(name, input_visual):
    '''
    input: images
    output: vector
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv3d(inputs=input_visual, filters=32, kernel_size=[1, 8, 8], strides=[1, 4, 4], padding='valid', activation=activation_fn, name='conv1', **initKernelAndBias)
        conv2 = tf.layers.conv3d(inputs=conv1, filters=64, kernel_size=[1, 4, 4], strides=[1, 2, 2], padding='valid', activation=activation_fn, name='conv2', **initKernelAndBias)
        conv3 = tf.layers.conv3d(inputs=conv2, filters=64, kernel_size=[1, 3, 3], strides=[1, 1, 1], padding='valid', activation=activation_fn, name='conv3', **initKernelAndBias)
        fc1 = tf.layers.dense(tf.layers.flatten(conv3), 256, activation_fn, name='fc1', **initKernelAndBias)
        return fc1


def actor_discrete(name, input_vector, out_shape, trainable=True, reuse=False):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''
    with tf.variable_scope(name):
        actor1 = tf.layers.dense(input_vector, 128, activation_fn, name='actor1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        actor2 = tf.layers.dense(actor1, 64, activation_fn, name='actor2', trainable=trainable, reuse=reuse, **initKernelAndBias)
        action_probs = tf.layers.dense(actor2, out_shape, tf.nn.softmax, name='action_probs', trainable=trainable, reuse=reuse, **initKernelAndBias)
        return action_probs


def actor_continuous(name, input_vector, out_shape, trainable=True, reuse=False):
    '''
    use for continuous action space.
    input: vector of state
    output: mean(mu) and variance(sigma) of Gaussian Distribution of actions given a state
    '''
    with tf.variable_scope(name):
        actor1 = tf.layers.dense(input_vector, 128, activation_fn, name='actor1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        actor2 = tf.layers.dense(actor1, 64, activation_fn, name='actor2', trainable=trainable, reuse=reuse, **initKernelAndBias)
        mu = tf.layers.dense(actor2, out_shape, tf.nn.tanh, name='mu', trainable=trainable, reuse=reuse, **initKernelAndBias)
        sigma1 = tf.layers.dense(actor1, 64, activation_fn, name='sigma1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        sigma = tf.layers.dense(sigma1, out_shape, tf.nn.sigmoid, name='sigma', trainable=trainable, reuse=reuse, **initKernelAndBias)
        return mu, sigma


def actor_dpg(name, input_vector, out_shape, trainable=True, reuse=False):
    '''
    use for DDPG and/or TD3 algorithms' actor network.
    input: vector of state
    output: deterministic action(mu) and disturbed action(action) given a state
    '''
    with tf.variable_scope(name):
        actor1 = tf.layers.dense(input_vector, 128, activation_fn, name='actor1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        actor2 = tf.layers.dense(actor1, 64, activation_fn, name='actor2', trainable=trainable, reuse=reuse, **initKernelAndBias)
        mu = tf.layers.dense(actor2, out_shape, tf.nn.tanh, name='mu', trainable=trainable, reuse=reuse, **initKernelAndBias)
        e = tf.random_normal(tf.shape(mu))
        action = tf.clip_by_value(mu + e, -1, 1)
        return mu, action


def critic_q_one(name, input_vector, trainable=True, reuse=False):
    '''
    use for evaluate the value given a state-action pair.
    input: tf.concat((state, action),axis = 1)
    output: q(s,a)
    '''
    with tf.variable_scope(name):
        critic1 = tf.layers.dense(input_vector, 256, activation_fn, name='critic1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        critic2 = tf.layers.dense(critic1, 256, activation_fn, name='critic2', trainable=trainable, reuse=reuse, **initKernelAndBias)
        q = tf.layers.dense(critic2, 1, None, name='q', trainable=trainable, reuse=reuse, **initKernelAndBias)
        # var = tf.get_variable_scope().global_variables()
        return q


def critic_v(name, input_vector, trainable=True, reuse=False):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''
    return critic_q_one(name, input_vector, trainable=trainable, reuse=reuse)


def critic_q_all(name, input_vector, out_shape, trainable=True, reuse=False):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''
    with tf.variable_scope(name):
        layer1 = tf.layers.dense(input_vector, 256, activation_fn, name='layer1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        layer2 = tf.layers.dense(layer1, 256, activation_fn, name='layer2', trainable=trainable, reuse=reuse, **initKernelAndBias)
        q = tf.layers.dense(layer2, out_shape, None, name='value', trainable=trainable, reuse=reuse, **initKernelAndBias)
        return q


def a_c_v_discrete(name, input_vector, out_shape, trainable=True, reuse=False):
    '''
    combine actor network and critic network, share some nn layers. use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state, v(s)
    '''
    with tf.variable_scope(name):
        share1 = tf.layers.dense(input_vector, 512, activation_fn, name='share1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        share2 = tf.layers.dense(share1, 256, activation_fn, name='share2', trainable=trainable, reuse=reuse, **initKernelAndBias)
        actor1 = tf.layers.dense(share2, 128, activation_fn, name='actor1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        actor2 = tf.layers.dense(actor1, 64, activation_fn, name='actor2', trainable=trainable, reuse=reuse, **initKernelAndBias)
        action_probs = tf.layers.dense(actor2, out_shape, tf.nn.softmax, name='action_probs', trainable=trainable, reuse=reuse, **initKernelAndBias)
        critic1 = tf.layers.dense(share2, 128, activation_fn, name='critic1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        critic2 = tf.layers.dense(critic1, 64, activation_fn, name='critic2', trainable=trainable, reuse=reuse, **initKernelAndBias)
        value = tf.layers.dense(critic2, 1, None, name='value', trainable=trainable, reuse=reuse, **initKernelAndBias)
        return action_probs, value


def a_c_v_continuous(name, input_vector, out_shape, trainable=True, reuse=False):
    '''
    combine actor network and critic network, share some nn layers. use for continuous action space.
    input: vector of state
    output: mean(mu) and variance(sigma) of Gaussian Distribution of actions given a state, v(s)
    '''
    with tf.variable_scope(name):
        share1 = tf.layers.dense(input_vector, 512, activation_fn, name='share1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        share2 = tf.layers.dense(share1, 256, activation_fn, name='share2', trainable=trainable, reuse=reuse, **initKernelAndBias)
        actor1 = tf.layers.dense(share2, 128, activation_fn, name='actor1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        actor2 = tf.layers.dense(actor1, 64, activation_fn, name='actor2', trainable=trainable, reuse=reuse, **initKernelAndBias)
        mu = tf.layers.dense(actor2, out_shape, tf.nn.tanh, name='mu', trainable=trainable, reuse=reuse, **initKernelAndBias)
        sigma1 = tf.layers.dense(actor1, 64, activation_fn, name='sigma1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        sigma = tf.layers.dense(sigma1, out_shape, tf.nn.sigmoid, name='sigma', trainable=trainable, reuse=reuse, **initKernelAndBias)
        critic1 = tf.layers.dense(share2, 128, activation_fn, name='critic1', trainable=trainable, reuse=reuse, **initKernelAndBias)
        critic2 = tf.layers.dense(critic1, 64, activation_fn, name='critic2', trainable=trainable, reuse=reuse, **initKernelAndBias)
        value = tf.layers.dense(critic2, 1, None, name='value', trainable=trainable, reuse=reuse, **initKernelAndBias)
        return mu, sigma, value
