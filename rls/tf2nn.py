import tensorflow as tf
from .activations import swish, mish
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model as M
from tensorflow.keras import Input as I
from rls.layers import Noisy, mlp


initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0.0, .1),
    'bias_initializer': tf.constant_initializer(0.1)    # 2.x 不需要指定dtype
}


class actor_dpg(M):
    '''
    use for DDPG and/or TD3 algorithms' actor network.
    input: vector of state
    output: deterministic action(mu) and disturbed action(action) given a state
    '''

    def __init__(self, vector_dim, output_shape, hidden_units, out_activation='tanh'):
        super().__init__()
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation=out_activation)
        self(I(shape=vector_dim))

    def call(self, x):
        mu = self.net(x)
        return mu


class actor_mu(M):
    '''
    use for PPO/PG algorithms' actor network.
    input: vector of state
    output: stochastic action(mu), normally is the mean of a Normal distribution
    '''

    def __init__(self, vector_dim, output_shape, hidden_units):
        super().__init__()
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation='tanh')
        self(I(shape=vector_dim))

    def call(self, x):
        mu = self.net(x)
        return mu


class actor_continuous(M):
    '''
    use for continuous action space.
    input: vector of state
    output: mean(mu) and log_variance(log_std) of Gaussian Distribution of actions given a state
    '''

    def __init__(self, vector_dim, output_shape, hidden_units):
        super().__init__()
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.mu = mlp(hidden_units['mu'], output_shape=output_shape, out_activation=None)
        self.log_std = mlp(hidden_units['log_std'], output_shape=output_shape, out_activation='tanh')
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        return (mu, log_std)


class actor_discrete(M):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''

    def __init__(self, vector_dim, output_shape, hidden_units):
        super().__init__()
        self.logits = mlp(hidden_units, output_shape=output_shape, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        logits = self.logits(x)
        return logits


class critic_q_one(M):
    '''
    use for evaluate the value given a state-action pair.
    input: tf.concat((state, action),axis = 1)
    output: q(s,a)
    '''

    def __init__(self, vector_dim, action_dim, hidden_units):
        super().__init__()
        self.net = mlp(hidden_units, output_shape=1, out_activation=None)
        self(I(shape=vector_dim), I(shape=action_dim))

    def call(self, x, a):
        q = self.net(tf.concat((x, a), axis=-1))
        return q


class critic_q_one2(M):
    '''
    Original architecture in DDPG paper.
    s-> layer -> feature, then tf.concat(feature, a) -> layer -> output
    '''

    def __init__(self, vector_dim, action_dim, hidden_units):
        assert len(hidden_units) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__()
        self.state_feature_net = mlp(hidden_units[0:1])
        self.net = mlp(hidden_units[1:], output_shape=1, out_activation=None)
        self(I(shape=vector_dim), I(shape=action_dim))

    def call(self, x, a):
        features = self.state_feature_net(x)
        q = self.net(tf.concat((x, action), axis=-1))
        return q


class critic_q_one3(M):
    '''
    Original architecture in TD3 paper.
    tf.concat(s,a) -> layer -> feature, then tf.concat(feature, a) -> layer -> output
    '''

    def __init__(self, vector_dim, action_dim, hidden_units):
        assert len(hidden_units) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__()
        self.feature_net = mlp(hidden_units[0:1])
        self.net = mlp(hidden_units[1:], output_shape=1, out_activation=None)
        self(I(shape=vector_dim), I(shape=action_dim))

    def call(self, x, a):
        features = self.feature_net(tf.concat((x, a), axis=-1))
        q = self.net(tf.concat((features, a), axis=-1))
        return q


class critic_v(M):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''

    def __init__(self, vector_dim, hidden_units):
        super().__init__()
        self.net = mlp(hidden_units, output_shape=1, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        v = self.net(x)
        return v


class critic_q_all(M):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''

    def __init__(self, vector_dim, output_shape, hidden_units, out_activation=None):
        super().__init__()
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation=out_activation)
        self(I(shape=vector_dim))

    def call(self, x):
        q = self.net(x)
        return q


class critic_q_bootstrap(M):
    '''
    use for bootstrapped dqn.
    '''

    def __init__(self, vector_dim, output_shape, head_num, hidden_units):
        super().__init__()
        self.nets = [mlp(hidden_units, output_shape=output_shape, out_activation=None) for _ in range(head_num)]
        self(I(shape=vector_dim))

    def call(self, x):
        q = tf.stack([net(x) for net in self.nets])  # [H, B, A]
        return q


class critic_dueling(M):
    '''
    Neural network for dueling deep Q network.
    Input:
        states: [batch_size, state_dim]
    Output:
        state value: [batch_size, 1]
        advantage: [batch_size, action_number]
    '''

    def __init__(self, vector_dim, output_shape, hidden_units):
        super().__init__()
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.v = mlp(hidden_units['v'], output_shape=1, out_activation=None)
        self.adv = mlp(hidden_units['adv'], output_shape=output_shape, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        v = self.v(x)    # [B, 1]
        adv = self.adv(x)  # [B, A]
        q = v + adv - tf.reduce_mean(adv, axis=1, keepdims=True)  # [B, A]
        return q


class oc_intra_option(M):
    '''
    Intra Option Neural network of Option-Critic.
    '''

    def __init__(self, vector_dim, output_shape, options_num, hidden_units, out_activation=None):
        super().__init__()
        self.actions_num = output_shape
        self.options_num = options_num
        self.pi = mlp(hidden_units, output_shape=options_num * output_shape, out_activation=out_activation)
        self(I(shape=vector_dim))

    def call(self, x):
        pi = self.pi(x)  # [B, P*A]
        pi = tf.reshape(pi, [-1, self.options_num, self.actions_num])  # [B, P*A] => [B, P, A]
        return pi


class aoc_share(M):
    '''
    Neural network for AOC.
    '''

    def __init__(self, vector_dim, action_dim, options_num, hidden_units, is_continuous=True):
        super().__init__()
        self.actions_num = action_dim
        self.options_num = options_num
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.q = mlp(hidden_units['q'], output_shape=options_num, out_activation=None)
        self.pi = mlp(hidden_units['intra_option'], output_shape=options_num * action_dim, out_activation='tanh' if is_continuous else None)
        self.beta = mlp(hidden_units['termination'], output_shape=options_num, out_activation='sigmoid')
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        q = self.q(x)   # [B, P]
        pi = self.pi(x)  # [B, P*A]
        pi = tf.reshape(pi, [-1, self.options_num, self.actions_num])  # [B, P*A] => [B, P, A]
        beta = self.beta(x)  # [B, P]
        return q, pi, beta


class ppoc_share(M):
    '''
    Neural network for PPOC.
    '''

    def __init__(self, vector_dim, action_dim, options_num, hidden_units, is_continuous=True):
        super().__init__()
        self.actions_num = action_dim
        self.options_num = options_num
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.q = mlp(hidden_units['q'], output_shape=options_num, out_activation=None)
        self.pi = mlp(hidden_units['intra_option'], output_shape=options_num * action_dim, out_activation='tanh' if is_continuous else None)
        self.beta = mlp(hidden_units['termination'], output_shape=options_num, out_activation='sigmoid')
        self.o = mlp(hidden_units['o'], output_shape=options_num, out_activation=tf.nn.log_softmax)
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        q = self.q(x)   # [B, P]
        pi = self.pi(x)  # [B, P*A]
        pi = tf.reshape(pi, [-1, self.options_num, self.actions_num])  # [B, P*A] => [B, P, A]
        beta = self.beta(x)  # [B, P]
        o = self.o(x)  # [B, P]
        return q, pi, beta, o


class a_c_v_continuous(M):
    '''
    combine actor network and critic network, share some nn layers. use for continuous action space.
    input: vector of state
    output: mean(mu) of Gaussian Distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, output_shape, hidden_units):
        super().__init__()
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.mu = mlp(hidden_units['mu'], output_shape=output_shape, out_activation='tanh')
        self.v = mlp(hidden_units['v'], output_shape=1, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        mu = self.mu(x)
        v = self.v(x)
        return (mu, v)


class a_c_v_discrete(M):
    '''
    combine actor network and critic network, share some nn layers. use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, output_shape, hidden_units):
        super().__init__()
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.logits = mlp(hidden_units['logits'], output_shape=output_shape, out_activation=None)
        self.v = mlp(hidden_units['v'], output_shape=1, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        logits = self.logits(x)
        v = self.v(x)
        return (logits, v)


class c51_distributional(M):
    '''
    neural network for C51
    '''

    def __init__(self, vector_dim, action_dim, atoms, hidden_units):
        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.net = mlp(hidden_units, output_shape=atoms * action_dim, out_activation='softmax')
        self(I(shape=vector_dim))

    def call(self, x):
        q_dist = self.net(x)    # [B, A*N]
        q_dist = tf.reshape(q_dist, [-1, self.action_dim, self.atoms])   # [B, A, N]
        return q_dist


class qrdqn_distributional(M):
    '''
    neural network for QRDQN
    '''

    def __init__(self, vector_dim, action_dim, nums, hidden_units):
        super().__init__()
        self.action_dim = action_dim
        self.nums = nums
        self.net = mlp(hidden_units, output_shape=nums * action_dim, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        q_dist = self.net(x)    # [B, A*N]
        q_dist = tf.reshape(q_dist, [-1, self.action_dim, self.nums])   # [B, A, N]
        return q_dist


class rainbow_dueling(M):
    '''
    Neural network for Rainbow.
    Input:
        states: [batch_size, state_dim]
    Output:
        state value: [batch_size, atoms]
        advantage: [batch_size, action_number * atoms]
    '''

    def __init__(self, vector_dim, action_dim, atoms, hidden_units):
        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.share = mlp(hidden_units['share'], layer=Noisy, out_layer=False)
        self.v = mlp(hidden_units['v'], layer=Noisy, output_shape=atoms, out_activation=None)
        self.adv = mlp(hidden_units['adv'], layer=Noisy, output_shape=action_dim * atoms, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        v = self.v(x)    # [B, N]
        adv = self.adv(x)   # [B, A*N]
        adv = tf.reshape(adv, [-1, self.action_dim, self.atoms])   # [B, A, N]
        adv -= tf.reduce_mean(adv)  # [B, A, N]
        adv = tf.transpose(adv, [1, 0, 2])  # [A, B, N]
        q = tf.transpose(v + adv, [1, 0, 2])    # [B, A, N]
        q = tf.nn.softmax(q)    # [B, A, N]
        return q  # [B, A, N]


class iqn_net(M):
    def __init__(self, vector_dim, action_dim, quantiles_idx, hidden_units):
        super().__init__()
        self.action_dim = action_dim
        self.q_net_head = mlp(hidden_units['q_net'], out_layer=False)   # [B, vector_dim]
        self.quantile_net = mlp(hidden_units['quantile'], out_layer=False)  # [N*B, quantiles_idx]
        self.q_net_tile = mlp(hidden_units['tile'], output_shape=action_dim, out_activation=None)   # [N*B, hidden_units['quantile'][-1]]
        self(I(shape=vector_dim), I(shape=quantiles_idx))

    def call(self, x, quantiles_tiled, *, quantiles_num=8):
        q_h = self.q_net_head(x)  # [B, obs_dim] => [B, h]
        q_h = tf.tile(q_h, [quantiles_num, 1])  # [B, h] => [N*B, h]
        quantile_h = self.quantile_net(quantiles_tiled)  # [N*B, quantiles_idx] => [N*B, h]
        hh = q_h * quantile_h  # [N*B, h]
        quantiles_value = self.q_net_tile(hh)  # [N*B, h] => [N*B, A]
        quantiles_value = tf.reshape(quantiles_value, (quantiles_num, -1, self.action_dim))   # [N*B, A] => [N, B, A]
        q = tf.reduce_mean(quantiles_value, axis=0)  # [N, B, A] => [B, A]
        return (quantiles_value, q)
