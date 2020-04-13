import tensorflow as tf
from .activations import swish, mish
from tensorflow.keras.layers import Dense
from Nn.layers import Noisy, mlp

activation_fn = 'tanh'

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0.0, .1),
    'bias_initializer': tf.constant_initializer(0.1)    # 2.x 不需要指定dtype
}


class actor_dpg(tf.keras.Model):
    '''
    use for DDPG and/or TD3 algorithms' actor network.
    input: vector of state
    output: deterministic action(mu) and disturbed action(action) given a state
    '''

    def __init__(self, vector_dim, output_shape, hidden_units):
        super().__init__()
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation='tanh')
        self(tf.keras.Input(shape=vector_dim))

    def call(self, x):
        mu = self.net(x)
        return mu


class actor_mu(tf.keras.Model):
    '''
    use for PPO/PG algorithms' actor network.
    input: vector of state
    output: stochastic action(mu), normally is the mean of a Normal distribution
    '''

    def __init__(self, vector_dim, output_shape, hidden_units):
        super().__init__()
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation='tanh')
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input)

    def call(self, x):
        mu = self.net(x)
        return mu


class actor_continuous(tf.keras.Model):
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
        self(tf.keras.Input(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        return (mu, log_std)


class actor_discrete(tf.keras.Model):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''

    def __init__(self, vector_dim, output_shape, hidden_units):
        super().__init__()
        self.logits = mlp(hidden_units, output_shape=output_shape, out_activation=None)
        self(tf.keras.Input(shape=vector_dim))

    def call(self, x):
        logits = self.logits(x)
        return logits


class critic_q_one(tf.keras.Model):
    '''
    use for evaluate the value given a state-action pair.
    input: tf.concat((state, action),axis = 1)
    output: q(s,a)
    '''

    def __init__(self, vector_dim, action_dim, hidden_units):
        super().__init__()
        self.net = mlp(hidden_units, output_shape=1, out_activation=None)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=action_dim))

    def call(self, x, a):
        q = self.net(tf.concat((x, a), axis=-1))
        return q


class critic_q_one2(tf.keras.Model):
    '''
    Original architecture in DDPG paper.
    s-> layer -> feature, then tf.concat(feature, a) -> layer -> output
    '''

    def __init__(self, vector_dim, action_dim, hidden_units):
        assert len(hidden_units) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__()
        self.state_feature_net = mlp(hidden_units[0:1])
        self.net = mlp(hidden_units[1:], output_shape=1, out_activation=None)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=action_dim))

    def call(self, x, a):
        features = self.state_feature_net(x)
        q = self.net(tf.concat((x, action), axis=-1))
        return q


class critic_q_one3(tf.keras.Model):
    '''
    Original architecture in TD3 paper.
    tf.concat(s,a) -> layer -> feature, then tf.concat(feature, a) -> layer -> output
    '''

    def __init__(self, vector_dim, action_dim, hidden_units):
        assert len(hidden_units) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__()
        self.feature_net = mlp(hidden_units[0:1])
        self.net = mlp(hidden_units[1:], output_shape=1, out_activation=None)
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=action_dim))

    def call(self, x, a):
        features = self.feature_net(tf.concat((x, a), axis=-1))
        q = self.net(tf.concat((features, a), axis=-1))
        return q


class critic_v(tf.keras.Model):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''

    def __init__(self, vector_dim, hidden_units):
        super().__init__()
        self.net = mlp(hidden_units, output_shape=1, out_activation=None)
        self(tf.keras.Input(shape=vector_dim))

    def call(self, x):
        v = self.net(x)
        return v


class critic_q_all(tf.keras.Model):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''

    def __init__(self, vector_dim, output_shape, hidden_units):
        super().__init__()
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation=None)
        self(tf.keras.Input(shape=vector_dim))

    def call(self, x):
        q = self.net(x)
        return q


class critic_q_bootstrap(tf.keras.Model):
    '''
    use for bootstrapped dqn.
    '''

    def __init__(self, vector_dim, output_shape, head_num, hidden_units):
        super().__init__()
        self.nets = [mlp(hidden_units, output_shape=output_shape, out_activation=None) for _ in range(head_num)]
        self(tf.keras.Input(shape=vector_dim))

    def call(self, x):
        q = tf.stack([net(x) for net in self.nets]) # [H, B, A]
        return q


class critic_dueling(tf.keras.Model):
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
        self(tf.keras.Input(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        v = self.v(x)    # [B, 1]
        adv = self.adv(x)  # [B, A]
        q = v + adv - tf.reduce_mean(adv, axis=1, keepdims=True)  # [B, A]
        return q

class oc(tf.keras.Model):
    '''
    Neural network for option-critic.
    '''

    def __init__(self, vector_dim, output_shape, options_num, hidden_units):
        super().__init__()
        self.actions_num = output_shape
        self.options_num = options_num
        self.share = mlp(hidden_units, out_layer=False)
        self.q = mlp([], output_shape=options_num, out_activation=None)
        self.pi = mlp([], output_shape=options_num*output_shape, out_activation=None)
        self.beta = mlp([], output_shape=options_num, out_activation='sigmoid')
        self(tf.keras.Input(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        q = self.q(x)   # [B, P]
        pi = self.pi(x) # [B, P*A]
        pi = tf.reshape(pi, [-1, self.options_num, self.actions_num]) # B, P*A] => [B, P, A]
        beta = self.beta(x) # [B, P]
        return q, pi, beta


class a_c_v_continuous(tf.keras.Model):
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
        self(tf.keras.Input(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        mu = self.mu(x)
        v = self.v(x)
        return (mu, v)


class a_c_v_discrete(tf.keras.Model):
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
        self(tf.keras.Input(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        logits = self.logits(x)
        v = self.v(x)
        return (logits, v)


class c51_distributional(tf.keras.Model):
    '''
    neural network for C51
    '''

    def __init__(self, vector_dim, action_dim, atoms, hidden_units):
        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.net = mlp(hidden_units, output_shape=atoms * action_dim, out_activation='softmax')
        self(tf.keras.Input(shape=vector_dim))

    def call(self, x):
        q_dist = self.net(x)    # [B, A*N]
        q_dist = tf.reshape(q_dist, [-1, self.action_dim, self.atoms])   # [B, A, N]
        return q_dist

class qrdqn_distributional(tf.keras.Model):
    '''
    neural network for QRDQN
    '''

    def __init__(self, vector_dim, action_dim, nums, hidden_units):
        super().__init__()
        self.action_dim = action_dim
        self.nums = nums
        self.net = mlp(hidden_units, output_shape=nums * action_dim, out_activation=None)
        self(tf.keras.Input(shape=vector_dim))

    def call(self, x):
        q_dist = self.net(x)    # [B, A*N]
        q_dist = tf.reshape(q_dist, [-1, self.action_dim, self.nums])   # [B, A, N]
        return q_dist


class rainbow_dueling(tf.keras.Model):
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
        self(tf.keras.Input(shape=vector_dim))

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


class iqn_net(tf.keras.Model):
    def __init__(self, vector_dim, action_dim, quantiles_idx, hidden_units):
        super().__init__()
        self.action_dim = action_dim
        self.q_net_head = mlp(hidden_units['q_net'], out_layer=False)   # [B, vector_dim]
        self.quantile_net = mlp(hidden_units['quantile'], out_layer=False)  # [N*B, quantiles_idx]
        self.q_net_tile = mlp(hidden_units['tile'], output_shape=action_dim, out_activation=None)   # [N*B, hidden_units['quantile'][-1]]
        self(tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=quantiles_idx))

    def call(self, x, quantiles_tiled, *, quantiles_num=8):
        q_h = self.q_net_head(x)  # [B, obs_dim] => [B, h]
        q_h = tf.tile(q_h, [quantiles_num, 1])  # [B, h] => [N*B, h]
        quantile_h = self.quantile_net(quantiles_tiled)  # [N*B, quantiles_idx] => [N*B, h]
        hh = q_h * quantile_h  # [N*B, h]
        quantiles_value = self.q_net_tile(hh)  # [N*B, h] => [N*B, A]
        quantiles_value = tf.reshape(quantiles_value, (quantiles_num, -1, self.action_dim))   # [N*B, A] => [N, B, A]
        q = tf.reduce_mean(quantiles_value, axis=0)  # [N, B, A] => [B, A]
        return (quantiles_value, q)
