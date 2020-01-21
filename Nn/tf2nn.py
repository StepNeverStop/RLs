import tensorflow as tf
from .activations import swish, mish
from tensorflow.keras.layers import Dense
from Nn.layers import Noisy, mlp

activation_fn = 'tanh'

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0.0, .1),
    'bias_initializer': tf.constant_initializer(0.1)    # 2.x 不需要指定dtype
}


class Model(tf.keras.Model):
    def __init__(self, visual_net, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.visual_net = visual_net
        self.tv = []
        self.tv += self.visual_net.trainable_variables

    def call(self, vector_input, visual_input, *args, **kwargs):
        '''
        args: action, reward, done. etc ...
        '''
        features = self.visual_net(visual_input)
        ret = self.init_or_run(
            tf.concat((vector_input, features), axis=-1),
            *args,
            **kwargs)
        return ret

    def update_vars(self):
        self.tv += self.trainable_variables

    def init_or_run(self, x):
        raise NotImplementedError


class actor_dpg(Model):
    '''
    use for DDPG and/or TD3 algorithms' actor network.
    input: vector of state
    output: deterministic action(mu) and disturbed action(action) given a state
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation='tanh')
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        mu = self.net(x)
        return mu


class actor_mu(Model):
    '''
    use for PPO/PG algorithms' actor network.
    input: vector of state
    output: stochastic action(mu), normally is the mean of a Normal distribution
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation='tanh')
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        mu = self.net(x)
        return mu


class actor_continuous(Model):
    '''
    use for continuous action space.
    input: vector of state
    output: mean(mu) and log_variance(log_std) of Gaussian Distribution of actions given a state
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.mu = mlp(hidden_units['mu'], output_shape=output_shape, out_activation=None)
        self.log_std = mlp(hidden_units['log_std'], output_shape=output_shape, out_activation='tanh')
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        x = self.share(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        return (mu, log_std)


class actor_discrete(Model):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.logits = mlp(hidden_units, output_shape=output_shape, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        logits = self.logits(x)
        return logits


class critic_q_one(Model):
    '''
    use for evaluate the value given a state-action pair.
    input: tf.concat((state, action),axis = 1)
    output: q(s,a)
    '''

    def __init__(self, vector_dim, action_dim, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.net = mlp(hidden_units, output_shape=1, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim), tf.keras.Input(shape=action_dim))
        self.update_vars()

    def init_or_run(self, x, a):
        q = self.net(tf.concat((x, a), axis=-1))
        return q


class critic_q_one2(Model):
    '''
    Original architecture in DDPG paper.
    s-> layer -> feature, then tf.concat(feature, a) -> layer -> output
    '''

    def __init__(self, vector_dim, action_dim, name, hidden_units, *, visual_net):
        assert len(hidden_units) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__(visual_net, name=name)
        self.state_feature_net = mlp(hidden_units[0:1])
        self.net = mlp(hidden_units[1:], output_shape=1, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim), tf.keras.Input(shape=action_dim))
        self.update_vars()

    def init_or_run(self, x, a):
        features = self.state_feature_net(x)
        q = self.net(tf.concat((x, action), axis=-1))
        return q


class critic_q_one3(Model):
    '''
    Original architecture in TD3 paper.
    tf.concat(s,a) -> layer -> feature, then tf.concat(feature, a) -> layer -> output
    '''

    def __init__(self, vector_dim, action_dim, name, hidden_units, *, visual_net):
        assert len(hidden_units) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__(visual_net, name=name)
        self.feature_net = mlp(hidden_units[0:1])
        self.net = mlp(hidden_units[1:], output_shape=1, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim), tf.keras.Input(shape=action_dim))
        self.update_vars()

    def init_or_run(self, x, a):
        features = self.feature_net(tf.concat((x, a), axis=-1))
        q = self.net(tf.concat((features, a), axis=-1))
        return q


class critic_v(Model):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''

    def __init__(self, vector_dim, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.net = mlp(hidden_units, output_shape=1, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        v = self.net(x)
        return v


class critic_q_all(Model):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        q = self.net(x)
        return q


class drqn_critic_q_all(Model):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.masking = tf.keras.layers.Masking(mask_value=0.)
        self.lstm_net = tf.keras.layers.LSTM(hidden_units['lstm'], return_state=True, return_sequences=True)
        self.net = mlp(hidden_units['dense'], output_shape=output_shape, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=(None, vector_dim + self.visual_net.hdim)))
        self.update_vars()

    def init_or_run(self, x, initial_state=None):
        x = self.masking(x)
        if initial_state is not None:
            x, h, c = self.lstm_net(x, initial_state=initial_state)
        else:
            x, h, c = self.lstm_net(x)
        q = self.net(x)
        q = tf.reshape(q, (-1, q.shape[-1]))    # [B, T, 1] => [B*T, 1]
        return (q, [h, c])


class critic_dueling(Model):
    '''
    Neural network for dueling deep Q network.
    Input:
        states: [batch_size, state_dim]
    Output:
        state value: [batch_size, 1]
        advantage: [batch_size, action_number]
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.v = mlp(hidden_units['v'], output_shape=1, out_activation=None)
        self.adv = mlp(hidden_units['adv'], output_shape=output_shape, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        x = self.share(x)
        v = self.v(x)    # [B, 1]
        adv = self.adv(x)  # [B, A]
        q = v + adv - tf.reduce_mean(adv, axis=1, keepdims=True)  # [B, A]
        return q


class a_c_v_continuous(Model):
    '''
    combine actor network and critic network, share some nn layers. use for continuous action space.
    input: vector of state
    output: mean(mu) of Gaussian Distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.mu = mlp(hidden_units['mu'], output_shape=output_shape, out_activation='tanh')
        self.v = mlp(hidden_units['v'], output_shape=1, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        x = self.share(x)
        mu = self.mu(x)
        v = self.v(x)
        return (mu, v)


class a_c_v_discrete(Model):
    '''
    combine actor network and critic network, share some nn layers. use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.logits = mlp(hidden_units['logits'], output_shape=output_shape, out_activation=None)
        self.v = mlp(hidden_units['v'], output_shape=1, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        x = self.share(x)
        logits = self.logits(x)
        v = self.v(x)
        return (logits, v)


class c51_distributional(Model):
    '''
    neural network for C51
    '''

    def __init__(self, vector_dim, action_dim, atoms, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.action_dim = action_dim
        self.atoms = atoms
        self.net = mlp(hidden_units, output_shape=atoms * action_dim, out_activation='softmax')
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        q_dist = self.net(x)    # [B, A*N]
        q_dist = tf.reshape(q_dist, [-1, self.action_dim, self.atoms])   # [B, A, N]
        return q_dist

class qrdqn_distributional(Model):
    '''
    neural network for QRDQN
    '''

    def __init__(self, vector_dim, action_dim, nums, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.action_dim = action_dim
        self.nums = nums
        self.net = mlp(hidden_units, output_shape=nums * action_dim, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        q_dist = self.net(x)    # [B, A*N]
        q_dist = tf.reshape(q_dist, [-1, self.action_dim, self.nums])   # [B, A, N]
        return q_dist


class rainbow_dueling(Model):
    '''
    Neural network for Rainbow.
    Input:
        states: [batch_size, state_dim]
    Output:
        state value: [batch_size, atoms]
        advantage: [batch_size, action_number * atoms]
    '''

    def __init__(self, vector_dim, action_dim, atoms, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.action_dim = action_dim
        self.atoms = atoms
        self.share = mlp(hidden_units['share'], layer=Noisy, out_layer=False)
        self.v = mlp(hidden_units['v'], layer=Noisy, output_shape=atoms, out_activation=None)
        self.adv = mlp(hidden_units['adv'], layer=Noisy, output_shape=action_dim * atoms, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        x = self.share(x)
        v = self.v(x)    # [B, N]
        adv = self.adv(x)   # [B, A*N]
        adv = tf.reshape(adv, [-1, self.action_dim, self.atoms])   # [B, A, N]
        adv -= tf.reduce_mean(adv)  # [B, A, N]
        adv = tf.transpose(adv, [1, 0, 2])  # [A, B, N]
        q = tf.transpose(v + adv, [1, 0, 2])    # [B, A, N]
        q = tf.nn.softmax(q)    # [B, A, N]
        return q  # [B, A, N]


class iqn_net(Model):
    def __init__(self, vector_dim, action_dim, quantiles_idx, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.action_dim = action_dim
        self.q_net_head = mlp(hidden_units['q_net'], out_layer=False)   # [B, vector_dim]
        self.quantile_net = mlp(hidden_units['quantile'], out_layer=False)  # [N*B, quantiles_idx]
        self.q_net_tile = mlp(hidden_units['tile'], output_shape=action_dim, out_activation=None)   # [N*B, hidden_units['quantile'][-1]]
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim), tf.keras.Input(shape=quantiles_idx))
        self.update_vars()

    def init_or_run(self, x, quantiles_tiled, *, quantiles_num=8):
        q_h = self.q_net_head(x)  # [B, obs_dim] => [B, h]
        q_h = tf.tile(q_h, [quantiles_num, 1])  # [B, h] => [N*B, h]
        quantile_h = self.quantile_net(quantiles_tiled)  # [N*B, quantiles_idx] => [N*B, h]
        hh = q_h * quantile_h  # [N*B, h]
        quantiles_value = self.q_net_tile(hh)  # [N*B, h] => [N*B, A]
        quantiles_value = tf.reshape(quantiles_value, (quantiles_num, -1, self.action_dim))   # [N*B, A] => [N, B, A]
        q = tf.reduce_mean(quantiles_value, axis=0)  # [N, B, A] => [B, A]
        return (quantiles_value, q)
