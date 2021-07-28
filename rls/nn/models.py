#!/usr/bin/env python3
# encoding: utf-8

import torch as t

from torch.nn import (Linear,
                      Softmax,
                      Sequential)

from rls.nn.mlps import MLP
from rls.utils.torch_utils import clip_nn_log_std


class ActorDPG(t.nn.Module):
    '''
    use for DDPG and/or TD3 algorithms' actor network.
    input: vector of state
    output: deterministic action(mu) and disturbed action(action) given a state
    '''

    def __init__(self, vector_dim, output_shape, network_settings, out_act='tanh'):
        super().__init__()
        self.net = MLP(vector_dim, network_settings, output_shape=output_shape, out_act=out_act)

    def forward(self, x):
        return self.net(x)


class ActorMuLogstd(t.nn.Module):
    '''
    use for PPO/PG algorithms' actor network.
    input: vector of state
    output: [stochastic action(mu), log of std]
    '''

    def __init__(self, vector_dim, output_shape, network_settings):
        super().__init__()
        self.condition_sigma = network_settings['condition_sigma']
        self.log_std_min, self.log_std_max = network_settings['log_std_bound']

        self.share = MLP(vector_dim, network_settings['hidden_units'])
        if network_settings['hidden_units']:
            ins = network_settings['hidden_units'][-1]
        else:
            ins = vector_dim
        self.mu = MLP(ins, output_shape=output_shape, out_act='tanh')
        if self.condition_sigma:
            self.log_std = MLP(ins, [], output_shape=output_shape)
        else:
            self.log_std = -0.5 * t.nn.Parameter(t.ones((1, output_shape)), requires_grad=True)

    def forward(self, x):
        x = self.share(x)
        mu = self.mu(x)
        if self.condition_sigma:
            log_std = self.log_std(x)
        else:
            log_std = self.log_std
        log_std.clamp_(self.log_std_min, self.log_std_max)
        batch_size = mu.shape[0]
        if batch_size:
            log_std = log_std.repeat(batch_size, 1)  # [1, N] => [B, N]
        return (mu, log_std)


class ActorCts(t.nn.Module):
    '''
    use for continuous action space.
    input: vector of state
    output: mean(mu) and log_variance(log_std) of Gaussian Distribution of actions given a state
    '''

    def __init__(self, vector_dim, output_shape, network_settings):
        super().__init__()
        self.soft_clip = network_settings['soft_clip']
        self.log_std_min, self.log_std_max = network_settings['log_std_bound']
        self.share = MLP(vector_dim, network_settings['share'])
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = vector_dim
        self.mu = MLP(ins, network_settings['mu'], output_shape=output_shape)
        self.log_std = MLP(ins, network_settings['log_std'], output_shape=output_shape)

    def forward(self, x):
        x = self.share(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        if self.soft_clip:
            log_std.tanh_()
            log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
        else:
            log_std.clamp_(self.log_std_min, self.log_std_max)
        return (mu, log_std)


class ActorDct(t.nn.Module):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''

    def __init__(self, vector_dim, output_shape, network_settings):
        super().__init__()
        self.logits = MLP(vector_dim, network_settings, output_shape=output_shape)

    def forward(self, x):
        logits = self.logits(x)
        return logits


class CriticQvalueOne(t.nn.Module):
    '''
    use for evaluate the value given a state-action pair.
    input: t.cat((state, action),axis = 1)
    output: q(s,a)
    '''

    def __init__(self, vector_dim, action_dim, network_settings):
        super().__init__()
        self.net = MLP(vector_dim + action_dim, network_settings, output_shape=1)

    def forward(self, x, a):
        q = self.net(t.cat((x, a), -1))
        return q


class CriticQvalueOneDDPG(t.nn.Module):
    '''
    Original architecture in DDPG paper.
    s-> layer -> feature, then t.cat(feature, a) -> layer -> output
    '''

    def __init__(self, vector_dim, action_dim, network_settings):
        assert len(network_settings) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__()
        self.state_feature_net = MLP(vector_dim, network_settings[0:1])
        self.net = MLP(vector_dim + action_dim, network_settings[1:], output_shape=1)

    def forward(self, x, a):
        features = self.state_feature_net(x)
        q = self.net(t.cat((x, action), -1))
        return q


class CriticQvalueOneTD3(t.nn.Module):
    '''
    Original architecture in TD3 paper.
    t.cat(s,a) -> layer -> feature, then t.cat(feature, a) -> layer -> output
    '''

    def __init__(self, vector_dim, action_dim, network_settings):
        assert len(network_settings) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__()
        self.feature_net = MLP(vector_dim + action_dim, network_settings[0:1])
        ins = network_settings[-1] + action_dim
        self.net = MLP(ins, network_settings[1:], output_shape=1)

    def forward(self, x, a):
        features = self.feature_net(t.cat((x, a), -1))
        q = self.net(t.cat((features, a), -1))
        return q


class CriticValue(t.nn.Module):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''

    def __init__(self, vector_dim, network_settings):
        super().__init__()
        self.net = MLP(vector_dim, network_settings, output_shape=1)

    def forward(self, x):
        v = self.net(x)
        return v


class CriticQvalueAll(t.nn.Module):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''

    def __init__(self, vector_dim, output_shape, network_settings, out_act=None):
        super().__init__()
        self.net = MLP(vector_dim, network_settings, output_shape=output_shape, out_act=out_act)

    def forward(self, x):
        q = self.net(x)
        return q


class CriticQvalueBootstrap(t.nn.Module):
    '''
    use for bootstrapped dqn.
    '''

    def __init__(self, vector_dim, output_shape, head_num, network_settings):
        super().__init__()
        self.nets = t.nn.ModuleList([MLP(vector_dim, network_settings, output_shape=output_shape) for _ in range(head_num)])

    def forward(self, x):
        q = t.stack([net(x) for net in self.nets])  # [H, B, A]
        return q


class CriticDueling(t.nn.Module):
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
        self.share = MLP(vector_dim, network_settings['share'])
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = vector_dim
        self.v = MLP(ins, network_settings['v'], output_shape=1)
        self.adv = MLP(ins, network_settings['adv'], output_shape=output_shape)

    def forward(self, x):
        x = self.share(x)
        v = self.v(x)    # [B, 1]
        adv = self.adv(x)  # [B, A]
        q = v + adv - adv.mean(1, keepdim=True)  # [B, A]
        return q


class OcIntraOption(t.nn.Module):
    '''
    Intra Option Neural network of Option-Critic.
    '''

    def __init__(self, vector_dim, output_shape, options_num, network_settings, out_act=None):
        super().__init__()
        self.actions_num = output_shape
        self.options_num = options_num
        self.pi = MLP(vector_dim, network_settings, output_shape=options_num * output_shape, out_act=out_act)

    def forward(self, x):
        pi = self.pi(x)  # [B, P*A]
        pi = pi.view(-1, self.options_num, self.actions_num)  # [B, P*A] => [B, P, A]
        return pi


class AocShare(t.nn.Module):
    '''
    Neural network for AOC.
    '''

    def __init__(self, vector_dim, action_dim, options_num, network_settings, is_continuous=True):
        super().__init__()
        self.actions_num = action_dim
        self.options_num = options_num
        self.share = MLP(vector_dim, network_settings['share'])
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = vector_dim
        self.q = MLP(ins, network_settings['q'], output_shape=options_num)
        self.pi = MLP(ins, network_settings['intra_option'], output_shape=options_num * action_dim, out_act='tanh' if is_continuous else None)
        self.beta = MLP(ins, network_settings['termination'], output_shape=options_num, out_act='sigmoid')

    def forward(self, x):
        x = self.share(x)
        q = self.q(x)   # [B, P]
        pi = self.pi(x)  # [B, P*A]
        pi = pi.view(-1, self.options_num, self.actions_num)  # [B, P*A] => [B, P, A]
        beta = self.beta(x)  # [B, P]
        return q, pi, beta


class PpocShare(t.nn.Module):
    '''
    Neural network for PPOC.
    '''

    def __init__(self, vector_dim, action_dim, options_num, network_settings, is_continuous=True):
        super().__init__()
        self.actions_num = action_dim
        self.options_num = options_num
        self.share = MLP(vector_dim, network_settings['share'])
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = vector_dim
        self.q = MLP(ins, network_settings['q'], output_shape=options_num)
        self.pi = MLP(ins, network_settings['intra_option'], output_shape=options_num * action_dim, out_act='tanh' if is_continuous else None)
        self.beta = MLP(ins, network_settings['termination'], output_shape=options_num, out_act='sigmoid')
        self.o = MLP(ins, network_settings['o'], output_shape=options_num, out_act=t.nn.functional.log_softmax)

    def forward(self, x):
        x = self.share(x)
        q = self.q(x)   # [B, P]
        pi = self.pi(x)  # [B, P*A]
        pi = pi.view(-1, self.options_num, self.actions_num)  # [B, P*A] => [B, P, A]
        beta = self.beta(x)  # [B, P]
        o = self.o(x)  # [B, P]
        return q, pi, beta, o


class ActorCriticValueCts(t.nn.Module):
    '''
    combine actor network and critic network, share some nn layers. use for continuous action space.
    input: vector of state
    output: mean(mu) of Gaussian Distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, output_shape, network_settings):
        super().__init__()
        self.condition_sigma = network_settings['condition_sigma']
        self.log_std_min, self.log_std_max = network_settings['log_std_bound']

        self.share = MLP(vector_dim, network_settings['share'])
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = vector_dim
        self.mu_logstd_share = MLP(ins, network_settings['mu'])
        self.v = MLP(ins, network_settings['v'], output_shape=1)
        if network_settings['mu']:
            ins = network_settings['mu'][-1]
        self.mu = MLP(ins, [], output_shape=output_shape, out_act='tanh')
        if self.condition_sigma:
            self.log_std = MLP(ins, [], output_shape=output_shape)
        else:
            self.log_std = -0.5 * t.nn.Parameter(t.ones((1, output_shape)), requires_grad=True)

    def forward(self, x):
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
                log_std = log_std.repeat(batch_size, 1)  # [1, N] => [B, N]
        log_std.clamp_(self.log_std_min, self.log_std_max)
        return (mu, log_std, v)


class ActorCriticValueDct(t.nn.Module):
    '''
    combine actor network and critic network, share some nn layers. use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, output_shape, network_settings):
        super().__init__()
        self.share = MLP(vector_dim, network_settings['share'])
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = vector_dim
        self.logits = MLP(ins, network_settings['logits'], output_shape=output_shape)
        self.v = MLP(ins, network_settings['v'], output_shape=1)

    def forward(self, x):
        x = self.share(x)
        logits = self.logits(x)
        v = self.v(x)
        return (logits, v)


class C51Distributional(t.nn.Module):
    '''
    neural network for C51
    '''

    def __init__(self, vector_dim, action_dim, atoms, network_settings):
        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.net = MLP(vector_dim, network_settings)
        if network_settings:
            ins = network_settings[-1]
        else:
            ins = vector_dim
        self.outputs = []
        for _ in range(action_dim):
            self.outputs.append(
                Sequential(
                    Linear(ins, atoms),
                    Softmax(-1)
                )
            )

    def forward(self, x):
        feat = self.net(x)    # [B, A*N]
        outputs = [output(feat) for output in self.outputs]  # A * [B, N]
        q_dist = t.cat(outputs, -1).view(-1, self.action_dim, self.atoms)   # A * [B, N] => [B, A*N] => [B, A, N]
        return q_dist


class QrdqnDistributional(t.nn.Module):
    '''
    neural network for QRDQN
    '''

    def __init__(self, vector_dim, action_dim, nums, network_settings):
        super().__init__()
        self.action_dim = action_dim
        self.nums = nums
        self.net = MLP(vector_dim, network_settings, output_shape=nums * action_dim)

    def forward(self, x):
        q_dist = self.net(x)    # [B, A*N]
        q_dist = q_dist.view(-1, self.action_dim, self.nums)   # [B, A, N]
        return q_dist


class RainbowDueling(t.nn.Module):
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
        self.share = MLP(vector_dim, network_settings['share'], layer='noisy')
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = vector_dim
        self.v = MLP(ins, network_settings['v'], layer='noisy', output_shape=atoms)
        self.adv = MLP(ins, network_settings['adv'], layer='noisy', output_shape=action_dim * atoms)

    def forward(self, x):
        x = self.share(x)
        v = self.v(x)    # [B, N]
        adv = self.adv(x)   # [B, A*N]
        adv = adv.view(-1, self.action_dim, self.atoms)  # [B, A, N]
        adv -= adv.mean()  # [B, A, N]
        adv = adv.permute(1, 0, 2)  # [A, B, N]
        q = (v + adv).permute(1, 0, 2)    # [B, A, N]
        q = q.softmax(-1)    # [B, A, N]
        return q  # [B, A, N]


class IqnNet(t.nn.Module):
    def __init__(self, vector_dim, action_dim, quantiles_idx, network_settings):
        super().__init__()
        self.action_dim = action_dim
        self.q_net_head = MLP(vector_dim, network_settings['q_net'])   # [B, vector_dim]
        self.quantile_net = MLP(quantiles_idx, network_settings['quantile'])  # [N*B, quantiles_idx]
        if network_settings['quantile']:    # TODO
            ins = network_settings['quantile'][-1]
        else:
            ins = quantiles_idx
        self.q_net_tile = MLP(ins, network_settings['tile'], output_shape=action_dim)   # [N*B, network_settings['quantile'][-1]]

    def forward(self, x, quantiles_tiled, *, quantiles_num=8):
        q_h = self.q_net_head(x)  # [B, obs_dim] => [B, h]
        q_h = q_h.repeat(quantiles_num, 1)  # [B, h] => [N*B, h]
        quantile_h = self.quantile_net(quantiles_tiled)  # [N*B, quantiles_idx] => [N*B, h]
        hh = q_h * quantile_h  # [N*B, h]
        quantiles_value = self.q_net_tile(hh)  # [N*B, h] => [N*B, A]
        quantiles_value = quantiles_value.view(quantiles_num, -1, self.action_dim)   # [N*B, A] => [N, B, A]
        q = quantiles_value.mean(0)  # [N, B, A] => [B, A]
        return (quantiles_value, q)
