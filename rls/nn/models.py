#!/usr/bin/env python3
# encoding: utf-8

import torch as t

from torch.nn import (Linear,
                      Softmax,
                      Sequential)

from rls.nn.mlps import MLP
from rls.utils.torch_utils import clip_nn_log_std
from rls.nn.represent_nets import RepresentationNetwork


class BaseModel(t.nn.Module):

    def __init__(self, obs_spec, rep_net_params):
        super().__init__()
        self.rep_net = RepresentationNetwork(obs_spec=obs_spec,
                                             rep_net_params=rep_net_params)
        self._cell_state = None

    def repre(self, x, cell_state=None):
        x, self._cell_state = self.rep_net(x, cell_state=cell_state)
        return x

    def get_cell_state(self):
        return self._cell_state

    def forward(self, x):
        raise NotImplementedError


class ActorDPG(BaseModel):
    '''
    use for DDPG and/or TD3 algorithms' actor network.
    input: vector of state
    output: deterministic action(mu) and disturbed action(action) given a state
    '''

    def __init__(self, obs_spec, rep_net_params, output_shape, network_settings, out_act='tanh'):
        super().__init__(obs_spec, rep_net_params)
        self.net = MLP(self.rep_net.h_dim, network_settings,
                       output_shape=output_shape, out_act=out_act)

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        return self.net(x)


class ActorMuLogstd(BaseModel):
    '''
    use for PPO/PG algorithms' actor network.
    input: vector of state
    output: [stochastic action(mu), log of std]
    '''

    def __init__(self, obs_spec, rep_net_params, output_shape, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.condition_sigma = network_settings['condition_sigma']
        self.log_std_min, self.log_std_max = network_settings['log_std_bound']

        self.share = MLP(self.rep_net.h_dim, network_settings['hidden_units'])
        if network_settings['hidden_units']:
            ins = network_settings['hidden_units'][-1]
        else:
            ins = self.rep_net.h_dim
        self.mu = MLP(ins, [], output_shape=output_shape, out_act='tanh')
        if self.condition_sigma:
            self.log_std = MLP(ins, [], output_shape=output_shape)
        else:
            self.log_std = t.nn.Parameter(-0.5 * t.ones(output_shape))

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        x = self.share(x)
        mu = self.mu(x)
        if self.condition_sigma:
            log_std = self.log_std(x)   # [T, B, *] or [B, *]
        else:
            # TODO:
            log_std = self.log_std.repeat(
                mu.shape[:-1]+(1,))   # [T, B, *] or [B, *]
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return (mu, log_std)


class ActorCts(BaseModel):
    '''
    use for continuous action space.
    input: vector of state
    output: mean(mu) and log_variance(log_std) of Gaussian Distribution of actions given a state
    '''

    def __init__(self, obs_spec, rep_net_params, output_shape, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.soft_clip = network_settings['soft_clip']
        self.log_std_min, self.log_std_max = network_settings['log_std_bound']
        self.share = MLP(self.rep_net.h_dim, network_settings['share'])
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = self.rep_net.h_dim
        self.mu = MLP(ins, network_settings['mu'], output_shape=output_shape)
        self.log_std = MLP(
            ins, network_settings['log_std'], output_shape=output_shape)

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        x = self.share(x)   # [B, *] or [T, B, *]
        mu = self.mu(x)  # [B, *] or [T, B, *]
        log_std = self.log_std(x)   # [B, *] or [T, B, *]
        if self.soft_clip:
            log_std.tanh_()  # [B, *] or [T, B, *]
            log_std = clip_nn_log_std(
                log_std, self.log_std_min, self.log_std_max)  # [B, *] or [T, B, *]
        else:
            # [B, *] or [T, B, *]
            log_std.clamp_(self.log_std_min, self.log_std_max)
        return (mu, log_std)


class ActorDct(BaseModel):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''

    def __init__(self, obs_spec, rep_net_params, output_shape, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.logits = MLP(self.rep_net.h_dim, network_settings,
                          output_shape=output_shape)

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        logits = self.logits(x)  # [B, *] or [T, B, *]
        return logits


class CriticQvalueOne(BaseModel):
    '''
    use for evaluate the value given a state-action pair.
    input: t.cat((state, action),axis = 1)
    output: q(s,a)
    '''

    def __init__(self, obs_spec, rep_net_params, action_dim, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.net = MLP(self.rep_net.h_dim + action_dim,
                       network_settings, output_shape=1)

    def forward(self, x, a, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        q = self.net(t.cat((x, a), -1))  # [B, 1] or [T, B, 1]
        return q


class CriticQvalueOneDDPG(BaseModel):
    '''
    Original architecture in DDPG paper.
    s-> layer -> feature, then t.cat(feature, a) -> layer -> output
    '''

    def __init__(self, obs_spec, rep_net_params, action_dim, network_settings):
        assert len(
            network_settings) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__(obs_spec, rep_net_params)
        self.state_feature_net = MLP(self.rep_net.h_dim, network_settings[0:1])
        self.net = MLP(self.rep_net.h_dim + action_dim,
                       network_settings[1:], output_shape=1)

    def forward(self, x, a, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        features = self.state_feature_net(x)
        q = self.net(t.cat((x, a), -1))
        return q


class CriticQvalueOneTD3(BaseModel):
    '''
    Original architecture in TD3 paper.
    t.cat(s,a) -> layer -> feature, then t.cat(feature, a) -> layer -> output
    '''

    def __init__(self, obs_spec, rep_net_params, action_dim, network_settings):
        assert len(
            network_settings) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__(obs_spec, rep_net_params)
        self.feature_net = MLP(self.rep_net.h_dim +
                               action_dim, network_settings[0:1])
        ins = network_settings[-1] + action_dim
        self.net = MLP(ins, network_settings[1:], output_shape=1)

    def forward(self, x, a, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        features = self.feature_net(t.cat((x, a), -1))
        q = self.net(t.cat((features, a), -1))
        return q


class CriticValue(BaseModel):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''

    def __init__(self, obs_spec, rep_net_params, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.net = MLP(self.rep_net.h_dim, network_settings, output_shape=1)

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        v = self.net(x)  # [B, *] or [T, B, *]
        return v


class CriticQvalueAll(BaseModel):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''

    def __init__(self, obs_spec, rep_net_params, output_shape, network_settings, out_act=None):
        super().__init__(obs_spec, rep_net_params)
        self.net = MLP(self.rep_net.h_dim, network_settings,
                       output_shape=output_shape, out_act=out_act)

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        q = self.net(x)  # [B, *] or [T, B, *]
        return q


class CriticQvalueBootstrap(BaseModel):
    '''
    use for bootstrapped dqn.
    '''

    def __init__(self, obs_spec, rep_net_params, output_shape, head_num, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.nets = t.nn.ModuleList(
            [MLP(self.rep_net.h_dim, network_settings, output_shape=output_shape) for _ in range(head_num)])

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        # [H, T, B, A] or [H, B, A]
        q = t.stack([net(x) for net in self.nets], 0)
        return q


class CriticDueling(BaseModel):
    '''
    Neural network for dueling deep Q network.
    Input:
        states: [batch_size, state_dim]
    Output:
        state value: [batch_size, 1]
        advantage: [batch_size, action_number]
    '''

    def __init__(self, obs_spec, rep_net_params, output_shape, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.share = MLP(self.rep_net.h_dim, network_settings['share'])
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = self.rep_net.h_dim
        self.v = MLP(ins, network_settings['v'], output_shape=1)
        self.adv = MLP(ins, network_settings['adv'], output_shape=output_shape)

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        x = self.share(x)   # [B, *] or [T, B, *]
        v = self.v(x)    # [B, 1] or [T, B, 1]
        adv = self.adv(x)  # [B, *] or [T, B, *]
        q = v + adv - adv.mean(-1, keepdim=True)  # [B, *] or [T, B, *]
        return q


class OcIntraOption(BaseModel):
    '''
    Intra Option Neural network of Option-Critic.
    '''

    def __init__(self, obs_spec, rep_net_params, output_shape, options_num, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.actions_num = output_shape
        self.options_num = options_num
        self.pi = MLP(self.rep_net.h_dim, network_settings,
                      output_shape=options_num * output_shape)

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)    # [B, *] or [T, B, *]
        pi = self.pi(x)  # [B, P*A] or [T, B, P*A]
        _shape = pi.shape[:-1] + (self.options_num, self.actions_num)
        pi = pi.view(_shape)  # [B, P, A] or [T, B, P, A]
        return pi


class AocShare(BaseModel):
    '''
    Neural network for AOC.
    '''

    def __init__(self, obs_spec, rep_net_params, action_dim, options_num, network_settings, is_continuous=True):
        super().__init__(obs_spec, rep_net_params)
        self.actions_num = action_dim
        self.options_num = options_num
        self.share = MLP(self.rep_net.h_dim, network_settings['share'])
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = self.rep_net.h_dim
        self.q = MLP(ins, network_settings['q'], output_shape=options_num)
        self.pi = MLP(ins, network_settings['intra_option'], output_shape=options_num *
                      action_dim, out_act='tanh' if is_continuous else None)
        self.beta = MLP(
            ins, network_settings['termination'], output_shape=options_num, out_act='sigmoid')

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)    # [B, *] or [T, B, *]
        x = self.share(x)   # [B, *] or [T, B, *]
        q = self.q(x)   # [B, P] or [T, B, P]
        pi = self.pi(x)  # [B, P*A] or [T, B, P*A]
        _shape = pi.shape[:-1] + (self.options_num, self.actions_num)
        pi = pi.view(_shape)  # [B, P, A] or [T, B, P, A]
        beta = self.beta(x)  # [B, P] or [T, B, P]
        return q, pi, beta


class PpocShare(BaseModel):
    '''
    Neural network for PPOC.
    '''

    def __init__(self, obs_spec, rep_net_params, action_dim, options_num, network_settings, is_continuous=True):
        super().__init__(obs_spec, rep_net_params)
        self.actions_num = action_dim
        self.options_num = options_num
        self.share = MLP(self.rep_net.h_dim, network_settings['share'])
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = self.rep_net.h_dim
        self.q = MLP(ins, network_settings['q'], output_shape=options_num)
        self.pi = MLP(ins, network_settings['intra_option'], output_shape=options_num *
                      action_dim, out_act='tanh' if is_continuous else None)
        self.beta = MLP(
            ins, network_settings['termination'], output_shape=options_num, out_act='sigmoid')
        self.o = MLP(
            ins, network_settings['o'], output_shape=options_num, out_act='log_softmax')

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)    # [B, *] or [T, B, *]
        x = self.share(x)   # [B, *] or [T, B, *]
        q = self.q(x)   # [B, P] or [T, B, P]
        pi = self.pi(x)  # [B, P*A] or [T, B, P*A]
        _shape = pi.shape[:-1] + (self.options_num, self.actions_num)
        pi = pi.view(_shape)  # [B, P, A] or [T, B, P, A]
        beta = self.beta(x)  # [B, P] or [T, B, P]
        o = self.o(x)  # [B, P] or [T, B, P]
        return q, pi, beta, o


class ActorCriticValueCts(BaseModel):
    '''
    combine actor network and critic network, share some nn layers. use for continuous action space.
    input: vector of state
    output: mean(mu) of Gaussian Distribution of actions given a state, v(s)
    '''

    def __init__(self, obs_spec, rep_net_params, output_shape, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.condition_sigma = network_settings['condition_sigma']
        self.log_std_min, self.log_std_max = network_settings['log_std_bound']

        self.share = MLP(self.rep_net.h_dim, network_settings['share'])
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = self.rep_net.h_dim
        self.mu_logstd_share = MLP(ins, network_settings['mu'])
        self.v = MLP(ins, network_settings['v'], output_shape=1)
        if network_settings['mu']:
            ins = network_settings['mu'][-1]
        self.mu = MLP(ins, [], output_shape=output_shape, out_act='tanh')
        if self.condition_sigma:
            self.log_std = MLP(ins, [], output_shape=output_shape)
        else:
            self.log_std = t.nn.Parameter(-0.5 * t.ones(output_shape))

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        x = self.share(x)
        v = self.v(x)
        x_mu_logstd = self.mu_logstd_share(x)
        mu = self.mu(x_mu_logstd)
        if self.condition_sigma:
            log_std = self.log_std(x_mu_logstd)  # [T, B, *] or [B, *]
        else:
            log_std = self.log_std.repeat(
                mu.shape[:-1]+(1,))   # [T, B, *] or [B, *]
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return (mu, log_std, v)


class ActorCriticValueDct(BaseModel):
    '''
    combine actor network and critic network, share some nn layers. use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state, v(s)
    '''

    def __init__(self, obs_spec, rep_net_params, output_shape, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.share = MLP(self.rep_net.h_dim, network_settings['share'])
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = self.rep_net.h_dim
        self.logits = MLP(
            ins, network_settings['logits'], output_shape=output_shape)
        self.v = MLP(ins, network_settings['v'], output_shape=1)

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        x = self.share(x)
        logits = self.logits(x)
        v = self.v(x)
        return (logits, v)


class C51Distributional(BaseModel):
    '''
    neural network for C51
    '''

    def __init__(self, obs_spec, rep_net_params, action_dim, atoms, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.action_dim = action_dim
        self.atoms = atoms
        self.net = MLP(self.rep_net.h_dim, network_settings)
        if network_settings:
            ins = network_settings[-1]
        else:
            ins = self.rep_net.h_dim
        self.outputs = []
        for _ in range(action_dim):
            self.outputs.append(
                Sequential(
                    Linear(ins, atoms),
                    Softmax(-1)
                )
            )

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        feat = self.net(x)    # [B, *] or [T, B, *]
        # A * [B, *] or A * [T, B, *]
        outputs = [output(feat) for output in self.outputs]
        q_dist = t.stack(outputs, -1)  # [B, *, A] or [T, B, *, A]
        return q_dist


class QrdqnDistributional(BaseModel):
    '''
    neural network for QRDQN
    '''

    def __init__(self, obs_spec, rep_net_params, action_dim, nums, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.action_dim = action_dim
        self.nums = nums
        self.net = MLP(self.rep_net.h_dim, network_settings,
                       output_shape=nums * action_dim)

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)
        q_dist = self.net(x)    # [B, A*N] or [T, B, A*N]
        _shape = q_dist.shape[:-1] + (self.action_dim, self.nums)
        q_dist = q_dist.view(_shape)   # [B, A, N] or [T, B, A, N]
        return q_dist


class RainbowDueling(BaseModel):
    '''
    Neural network for Rainbow.
    Input:
        states: [batch_size, state_dim]
    Output:
        state value: [batch_size, atoms]
        advantage: [batch_size, action_number * atoms]
    '''

    def __init__(self, obs_spec, rep_net_params, action_dim, atoms, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.action_dim = action_dim
        self.atoms = atoms
        self.share = MLP(self.rep_net.h_dim,
                         network_settings['share'], layer='noisy')
        if network_settings['share']:
            ins = network_settings['share'][-1]
        else:
            ins = self.rep_net.h_dim
        self.v = MLP(ins, network_settings['v'],
                     layer='noisy', output_shape=atoms)
        self.adv = MLP(
            ins, network_settings['adv'], layer='noisy', output_shape=action_dim * atoms)

    def forward(self, x, cell_state=None):
        x = self.repre(x, cell_state=cell_state)    # [B, N] or [T, B, N]
        x = self.share(x)   # [B, N] or [T, B, N]
        v = self.v(x)       # [B, N] or [T, B, N]
        adv = self.adv(x)   # [B, A*N] or [T, B, A*N]
        adv -= adv.mean(-1, keepdim=True)  # [B, A*N] or [T, B, A*N]
        q = v.repeat((1,)*(v.ndim-1)+(self.action_dim,)) + \
            adv  # [B, A*N] or [T, B, A*N]
        # [B, A, N] or [T, B, A, N]
        q = q.view(q.shape[:-1]+(self.action_dim, self.atoms))
        q = q.softmax(-1)    # [B, A, N] or [T, B, A, N]
        return q  # [B, A, N] or [T, B, A, N]


class IqnNet(BaseModel):
    def __init__(self, obs_spec, rep_net_params, action_dim, quantiles_idx, network_settings):
        super().__init__(obs_spec, rep_net_params)
        self.action_dim = action_dim
        # [B, self.rep_net.h_dim]
        self.q_net_head = MLP(self.rep_net.h_dim, network_settings['q_net'])
        # [N*B, quantiles_idx]
        self.quantile_net = MLP(quantiles_idx, network_settings['quantile'])
        if network_settings['quantile']:    # TODO
            ins = network_settings['quantile'][-1]
        else:
            ins = quantiles_idx
        # [N*B, network_settings['quantile'][-1]]
        self.q_net_tile = MLP(
            ins, network_settings['tile'], output_shape=action_dim)

    def forward(self, x, quantiles_tiled, *, cell_state=None):
        '''
        params:
            x: [B, *] or [T, B, *]
            quantiles_tiled: [N*B, *] or [T, N*B, *]
        '''
        x = self.repre(x, cell_state=cell_state)    # [B, *] or [T, B, *]
        q_h = self.q_net_head(x)  # [B, *] or [T, B, *]

        quantiles_num = quantiles_tiled.shape[-2] // q_h.shape[-2]

        # [B, *] => [N*B, *] or [T, B, *] => [T, N*B, *]
        q_h = q_h.repeat((1,)*(q_h.ndim-2) + (quantiles_num, 1))
        quantile_h = self.quantile_net(
            quantiles_tiled)  # [N*B, *] or [T, N*B, *]
        hh = q_h * quantile_h  # [N*B, *] or [T, N*B, *]
        quantiles_value = self.q_net_tile(hh)  # [N*B, A] or [T, N*B, *]
        _shape = quantiles_value.shape[:-2] + \
            (quantiles_num, -1, self.action_dim)
        # [N*B, A] => [N, B, A]  or [T, N*B, A] => [T, N, B, A]
        quantiles_value = quantiles_value.view(_shape)
        return quantiles_value  # [N, B, A] or [T, N, B, A]


class MACriticQvalueOne(t.nn.Module):
    '''
    use for evaluate the value given a state-action pair.
    input: t.cat((state, action),axis = 1)
    output: q(s,a)
    '''

    def __init__(self, obs_specs, rep_net_params, action_dim, network_settings):
        super().__init__()
        self.rep_nets = t.nn.ModuleList()
        for obs_spec in obs_specs:
            self.rep_nets.append(RepresentationNetwork(
                obs_spec, rep_net_params
            ))
        h_dim = sum([rep_net.h_dim for rep_net in self.rep_nets])
        self.net = MLP(h_dim + action_dim, network_settings, output_shape=1)

    def forward(self, x, a):
        outs = []
        for _in, rep_net in zip(x, self.rep_nets):
            _out, _ = rep_net(_in)
            outs.append(_out)
        x = t.cat(outs, -1)
        q = self.net(t.cat((x, a), -1))
        return q
