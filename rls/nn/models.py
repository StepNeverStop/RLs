#!/usr/bin/env python3
# encoding: utf-8

import tensorflow as tf

from tensorflow.keras import Model as M
from tensorflow.keras import Input as I
from tensorflow.keras.layers import Dense

from rls.nn.layers import (Noisy,
                           mlp)
from rls.utils.specs import OutputNetworkType
from rls.utils.tf2_utils import clip_nn_log_std


def get_output_network_from_type(network_type: OutputNetworkType):
    OUTPUT_NETWORKS = {
        OutputNetworkType.ACTOR_DPG: ActorDPG,
        OutputNetworkType.ACTOR_MU: ActorMu,
        OutputNetworkType.ACTOR_MU_LOGSTD: ActorMuLogstd,
        OutputNetworkType.ACTOR_CTS: ActorCts,
        OutputNetworkType.ACTOR_DCT: ActorDct,
        OutputNetworkType.CRITIC_QVALUE_ONE: CriticQvalueOne,
        OutputNetworkType.CRITIC_QVALUE_ONE_DDPG: CriticQvalueOneDDPG,
        OutputNetworkType.CRITIC_QVALUE_ONE_TD3: CriticQvalueOneTD3,
        OutputNetworkType.CRITIC_VALUE: CriticValue,
        OutputNetworkType.CRITIC_QVALUE_ALL: CriticQvalueAll,
        OutputNetworkType.CRITIC_QVALUE_BOOTSTRAP: CriticQvalueBootstrap,
        OutputNetworkType.CRITIC_DUELING: CriticDueling,
        OutputNetworkType.OC_INTRA_OPTION: OcIntraOption,
        OutputNetworkType.AOC_SHARE: AocShare,
        OutputNetworkType.PPOC_SHARE: PpocShare,
        OutputNetworkType.ACTOR_CRITIC_VALUE_CTS: ActorCriticValueCts,
        OutputNetworkType.ACTOR_CRITIC_VALUE_DCT: ActorCriticValueDct,
        OutputNetworkType.C51_DISTRIBUTIONAL: C51Distributional,
        OutputNetworkType.QRDQN_DISTRIBUTIONAL: QrdqnDistributional,
        OutputNetworkType.RAINBOW_DUELING: RainbowDueling,
        OutputNetworkType.IQN_NET: IqnNet
    }
    return OUTPUT_NETWORKS.get(network_type, OUTPUT_NETWORKS[OutputNetworkType.ACTOR_CTS])


class ActorDPG(M):
    '''
    use for DDPG and/or TD3 algorithms' actor network.
    input: vector of state
    output: deterministic action(mu) and disturbed action(action) given a state
    '''

    def __init__(self, vector_dim, output_shape, network_settings, out_activation='tanh'):
        super().__init__()
        self.net = mlp(network_settings, output_shape=output_shape, out_activation=out_activation)
        self(I(shape=vector_dim))

    def call(self, x):
        mu = self.net(x)
        return mu


class ActorMu(M):
    '''
    input: vector of state
    output: stochastic action(mu), normally is the mean of a Normal distribution
    '''

    def __init__(self, vector_dim, output_shape, network_settings):
        super().__init__()
        self.net = mlp(network_settings, output_shape=output_shape, out_activation='tanh')
        self(I(shape=vector_dim))

    def call(self, x):
        mu = self.net(x)
        return mu


class ActorMuLogstd(M):
    '''
    use for PPO/PG algorithms' actor network.
    input: vector of state
    output: [stochastic action(mu), log of std]
    '''

    def __init__(self, vector_dim, output_shape, network_settings):
        super().__init__()
        self.condition_sigma = network_settings['condition_sigma']
        self.log_std_min, self.log_std_max = network_settings['log_std_bound']

        self.share = mlp(network_settings['hidden_units'], out_layer=False)
        self.mu = mlp([], output_shape=output_shape, out_activation='tanh')
        if self.condition_sigma:
            self.log_std = mlp([], output_shape=output_shape, out_activation=None)
        else:
            self.log_std = tf.Variable(initial_value=-0.5 * tf.ones(shape=(1, output_shape), dtype=tf.dtypes.float32), trainable=True)
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        mu = self.mu(x)
        if self.condition_sigma:
            log_std = self.log_std(x)
        else:
            log_std = self.log_std
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        batch_size = mu.shape[0]
        if batch_size:
            log_std = tf.tile(log_std, [batch_size, 1])  # [1, N] => [B, N]
        return (mu, log_std)


class ActorCts(M):
    '''
    use for continuous action space.
    input: vector of state
    output: mean(mu) and log_variance(log_std) of Gaussian Distribution of actions given a state
    '''

    def __init__(self, vector_dim, output_shape, network_settings):
        super().__init__()
        self.soft_clip = network_settings['soft_clip']
        self.log_std_min, self.log_std_max = network_settings['log_std_bound']
        self.share = mlp(network_settings['share'], out_layer=False)
        self.mu = mlp(network_settings['mu'], output_shape=output_shape, out_activation=None)
        self.log_std = mlp(network_settings['log_std'], output_shape=output_shape, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        if self.soft_clip:
            log_std = tf.tanh(log_std)
            log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
        else:
            log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        return (mu, log_std)


class ActorDct(M):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''

    def __init__(self, vector_dim, output_shape, network_settings):
        super().__init__()
        self.logits = mlp(network_settings, output_shape=output_shape, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        logits = self.logits(x)
        return logits


class CriticQvalueOne(M):
    '''
    use for evaluate the value given a state-action pair.
    input: tf.concat((state, action),axis = 1)
    output: q(s,a)
    '''

    def __init__(self, vector_dim, action_dim, network_settings):
        super().__init__()
        self.net = mlp(network_settings, output_shape=1, out_activation=None)
        self(I(shape=vector_dim), I(shape=action_dim))

    def call(self, x, a):
        q = self.net(tf.concat((x, a), axis=-1))
        return q


class CriticQvalueOneDDPG(M):
    '''
    Original architecture in DDPG paper.
    s-> layer -> feature, then tf.concat(feature, a) -> layer -> output
    '''

    def __init__(self, vector_dim, action_dim, network_settings):
        assert len(network_settings) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__()
        self.state_feature_net = mlp(network_settings[0:1])
        self.net = mlp(network_settings[1:], output_shape=1, out_activation=None)
        self(I(shape=vector_dim), I(shape=action_dim))

    def call(self, x, a):
        features = self.state_feature_net(x)
        q = self.net(tf.concat((x, action), axis=-1))
        return q


class CriticQvalueOneTD3(M):
    '''
    Original architecture in TD3 paper.
    tf.concat(s,a) -> layer -> feature, then tf.concat(feature, a) -> layer -> output
    '''

    def __init__(self, vector_dim, action_dim, network_settings):
        assert len(network_settings) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__()
        self.feature_net = mlp(network_settings[0:1])
        self.net = mlp(network_settings[1:], output_shape=1, out_activation=None)
        self(I(shape=vector_dim), I(shape=action_dim))

    def call(self, x, a):
        features = self.feature_net(tf.concat((x, a), axis=-1))
        q = self.net(tf.concat((features, a), axis=-1))
        return q


class CriticValue(M):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''

    def __init__(self, vector_dim, network_settings):
        super().__init__()
        self.net = mlp(network_settings, output_shape=1, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        v = self.net(x)
        return v


class CriticQvalueAll(M):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''

    def __init__(self, vector_dim, output_shape, network_settings, out_activation=None):
        super().__init__()
        self.net = mlp(network_settings, output_shape=output_shape, out_activation=out_activation)
        self(I(shape=vector_dim))

    def call(self, x):
        q = self.net(x)
        return q


class CriticQvalueBootstrap(M):
    '''
    use for bootstrapped dqn.
    '''

    def __init__(self, vector_dim, output_shape, head_num, network_settings):
        super().__init__()
        self.nets = [mlp(network_settings, output_shape=output_shape, out_activation=None) for _ in range(head_num)]
        self(I(shape=vector_dim))

    def call(self, x):
        q = tf.stack([net(x) for net in self.nets])  # [H, B, A]
        return q


class CriticDueling(M):
    '''
    Neural network for dueling deep Q network.
    Input:
        states: [batch_size, state_dim]
    Output:
        state value: [batch_size, 1]
        advantage: [batch_size, action_number]
    '''

    def __init__(self, vector_dim, output_shape, network_settings):
        super().__init__()
        self.share = mlp(network_settings['share'], out_layer=False)
        self.v = mlp(network_settings['v'], output_shape=1, out_activation=None)
        self.adv = mlp(network_settings['adv'], output_shape=output_shape, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        v = self.v(x)    # [B, 1]
        adv = self.adv(x)  # [B, A]
        q = v + adv - tf.reduce_mean(adv, axis=1, keepdims=True)  # [B, A]
        return q


class OcIntraOption(M):
    '''
    Intra Option Neural network of Option-Critic.
    '''

    def __init__(self, vector_dim, output_shape, options_num, network_settings, out_activation=None):
        super().__init__()
        self.actions_num = output_shape
        self.options_num = options_num
        self.pi = mlp(network_settings, output_shape=options_num * output_shape, out_activation=out_activation)
        self(I(shape=vector_dim))

    def call(self, x):
        pi = self.pi(x)  # [B, P*A]
        pi = tf.reshape(pi, [-1, self.options_num, self.actions_num])  # [B, P*A] => [B, P, A]
        return pi


class AocShare(M):
    '''
    Neural network for AOC.
    '''

    def __init__(self, vector_dim, action_dim, options_num, network_settings, is_continuous=True):
        super().__init__()
        self.actions_num = action_dim
        self.options_num = options_num
        self.share = mlp(network_settings['share'], out_layer=False)
        self.q = mlp(network_settings['q'], output_shape=options_num, out_activation=None)
        self.pi = mlp(network_settings['intra_option'], output_shape=options_num * action_dim, out_activation='tanh' if is_continuous else None)
        self.beta = mlp(network_settings['termination'], output_shape=options_num, out_activation='sigmoid')
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        q = self.q(x)   # [B, P]
        pi = self.pi(x)  # [B, P*A]
        pi = tf.reshape(pi, [-1, self.options_num, self.actions_num])  # [B, P*A] => [B, P, A]
        beta = self.beta(x)  # [B, P]
        return q, pi, beta


class PpocShare(M):
    '''
    Neural network for PPOC.
    '''

    def __init__(self, vector_dim, action_dim, options_num, network_settings, is_continuous=True):
        super().__init__()
        self.actions_num = action_dim
        self.options_num = options_num
        self.share = mlp(network_settings['share'], out_layer=False)
        self.q = mlp(network_settings['q'], output_shape=options_num, out_activation=None)
        self.pi = mlp(network_settings['intra_option'], output_shape=options_num * action_dim, out_activation='tanh' if is_continuous else None)
        self.beta = mlp(network_settings['termination'], output_shape=options_num, out_activation='sigmoid')
        self.o = mlp(network_settings['o'], output_shape=options_num, out_activation=tf.nn.log_softmax)
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        q = self.q(x)   # [B, P]
        pi = self.pi(x)  # [B, P*A]
        pi = tf.reshape(pi, [-1, self.options_num, self.actions_num])  # [B, P*A] => [B, P, A]
        beta = self.beta(x)  # [B, P]
        o = self.o(x)  # [B, P]
        return q, pi, beta, o


class ActorCriticValueCts(M):
    '''
    combine actor network and critic network, share some nn layers. use for continuous action space.
    input: vector of state
    output: mean(mu) of Gaussian Distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, output_shape, network_settings):
        super().__init__()
        self.condition_sigma = network_settings['condition_sigma']
        self.log_std_min, self.log_std_max = network_settings['log_std_bound']

        self.share = mlp(network_settings['share'], out_layer=False)
        self.mu_logstd_share = mlp(network_settings['mu'], out_layer=False)
        self.mu = mlp([], output_shape=output_shape, out_activation='tanh')
        self.v = mlp(network_settings['v'], output_shape=1, out_activation=None)
        if self.condition_sigma:
            self.log_std = mlp([], output_shape=output_shape, out_activation=None)
        else:
            self.log_std = tf.Variable(initial_value=-0.5 * tf.ones(shape=(1, output_shape), dtype=tf.dtypes.float32), trainable=True)
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        v = self.v(x)
        x_mu_logstd = self.mu_logstd_share(x)
        mu = self.mu(x_mu_logstd)
        if self.condition_sigma:
            log_std = self.log_std(x_mu_logstd)
        else:
            log_std = self.log_std
            batch_size = mu.shape[0]
            if batch_size:
                log_std = tf.tile(log_std, [batch_size, 1])  # [1, N] => [B, N]
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        return (mu, log_std, v)


class ActorCriticValueDct(M):
    '''
    combine actor network and critic network, share some nn layers. use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, output_shape, network_settings):
        super().__init__()
        self.share = mlp(network_settings['share'], out_layer=False)
        self.logits = mlp(network_settings['logits'], output_shape=output_shape, out_activation=None)
        self.v = mlp(network_settings['v'], output_shape=1, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        x = self.share(x)
        logits = self.logits(x)
        v = self.v(x)
        return (logits, v)


class C51Distributional(M):
    '''
    neural network for C51
    '''

    def __init__(self, vector_dim, action_dim, atoms, network_settings):
        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.net = mlp(network_settings, out_layer=False)
        self.outputs = []
        for _ in range(action_dim):
            self.outputs.append(Dense(atoms, activation='softmax'))
        self(I(shape=vector_dim))

    def call(self, x):
        feat = self.net(x)    # [B, A*N]
        outputs = [output(feat) for output in self.outputs]  # A * [B, N]
        q_dist = tf.reshape(tf.concat(outputs, axis=-1), [-1, self.action_dim, self.atoms])   # A * [B, N] => [B, A*N] => [B, A, N]
        return q_dist


class QrdqnDistributional(M):
    '''
    neural network for QRDQN
    '''

    def __init__(self, vector_dim, action_dim, nums, network_settings):
        super().__init__()
        self.action_dim = action_dim
        self.nums = nums
        self.net = mlp(network_settings, output_shape=nums * action_dim, out_activation=None)
        self(I(shape=vector_dim))

    def call(self, x):
        q_dist = self.net(x)    # [B, A*N]
        q_dist = tf.reshape(q_dist, [-1, self.action_dim, self.nums])   # [B, A, N]
        return q_dist


class RainbowDueling(M):
    '''
    Neural network for Rainbow.
    Input:
        states: [batch_size, state_dim]
    Output:
        state value: [batch_size, atoms]
        advantage: [batch_size, action_number * atoms]
    '''

    def __init__(self, vector_dim, action_dim, atoms, network_settings):
        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.share = mlp(network_settings['share'], layer=Noisy, out_layer=False)
        self.v = mlp(network_settings['v'], layer=Noisy, output_shape=atoms, out_activation=None)
        self.adv = mlp(network_settings['adv'], layer=Noisy, output_shape=action_dim * atoms, out_activation=None)
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


class IqnNet(M):
    def __init__(self, vector_dim, action_dim, quantiles_idx, network_settings):
        super().__init__()
        self.action_dim = action_dim
        self.q_net_head = mlp(network_settings['q_net'], out_layer=False)   # [B, vector_dim]
        self.quantile_net = mlp(network_settings['quantile'], out_layer=False)  # [N*B, quantiles_idx]
        self.q_net_tile = mlp(network_settings['tile'], output_shape=action_dim, out_activation=None)   # [N*B, network_settings['quantile'][-1]]
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
