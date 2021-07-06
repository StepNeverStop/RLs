

import numpy as np
import tensorflow as tf

from typing import (List,
                    Dict)
from copy import deepcopy
from abc import ABC, abstractmethod
from tensorflow.keras import Model as M

from rls.utils.specs import OutputNetworkType
from rls.nn.networks import get_visual_network_from_type
from rls.nn.models import get_output_network_from_type
from rls.nn.networks import (MultiVectorNetwork,
                             MultiVisualNetwork,
                             EncoderNetwork,
                             MemoryNetwork)
from rls.utils.logging_utils import get_logger
from rls.utils.specs import (ObsSpec,
                             VectorNetworkType,
                             VisualNetworkType,
                             MemoryNetworkType)
from rls.utils.tf2_utils import update_target_net_weights
logger = get_logger(__name__)


class RepresentationNetwork(ABC):

    def __init__(self, name: str = 'test'):
        self.name = name
        self.h_dim = None

    @abstractmethod
    def __call__(self):
        pass

    @property
    @abstractmethod
    def trainable_variables(self):
        pass

    @property
    @abstractmethod
    def weights(self):
        pass

    @property
    @abstractmethod
    def _policy_models(self):
        pass

    @property
    @abstractmethod
    def _all_models(self):
        pass

    @abstractmethod
    def _copy(self):
        pass


class DefaultRepresentationNetwork(RepresentationNetwork):
    '''
      visual -> visual_net -> feat ↘
                                     feat -> encoder_net -> feat ↘                ↗ feat
      vector -> vector_net -> feat ↗                             -> memory_net ->
                                                      cell_state ↗                ↘ cell_state
    '''

    def __init__(self,
                 name: str,
                 obs_spec: ObsSpec,
                 representation_net_params: Dict):
        super().__init__(name)

        self.obs_spec = obs_spec
        self.representation_net_params = representation_net_params

        vector_net_params = dict(representation_net_params.get('vector_net_params', {}))
        visual_net_params = dict(representation_net_params.get('visual_net_params', {}))

        vector_net_params['network_type'] = VectorNetworkType(vector_net_params['network_type'])
        visual_net_params['network_type'] = VisualNetworkType(visual_net_params['network_type'])

        self.vector_net = MultiVectorNetwork(obs_spec.vector_dims, **vector_net_params)
        logger.debug('initialize vector network successfully.')
        self.visual_net = MultiVisualNetwork(obs_spec.visual_dims, **visual_net_params)
        logger.debug('initialize visual network successfully.')

        self.h_dim = self.vector_net.h_dim + self.visual_net.h_dim
        self.use_encoder = bool(representation_net_params.get('use_encoder', False))
        if self.use_encoder:
            encoder_net_params = dict(representation_net_params.get('encoder_net_params', {}))
            self.encoder_net = EncoderNetwork(self.h_dim, **encoder_net_params)
            logger.debug('initialize encoder network successfully.')
            self.h_dim = self.encoder_net.h_dim

        self.use_rnn = bool(representation_net_params.get('use_rnn', False))
        if self.use_rnn:
            memory_net_params = dict(representation_net_params.get('memory_net_params', {}))
            memory_net_params['network_type'] = MemoryNetworkType(memory_net_params['network_type'])
            self.memory_net = MemoryNetwork(self.h_dim, **memory_net_params)
            logger.debug('initialize memory network successfully.')
            self.h_dim = self.memory_net.h_dim

    @tf.function
    def __call__(self, obs, cell_state):
        '''
        params:
            cell_state: Tuple([B, z],)
        return:
            feat: [B, a]
            cell_state: Tuple([B, z],)
        '''
        if self.obs_spec.has_vector_observation and self.obs_spec.has_visual_observation:
            vec_feat = self.vector_net(*obs.vector)
            vis_feat = self.visual_net(*obs.visual)
            feat = tf.concat([vec_feat, vis_feat], axis=-1)
        elif self.obs_spec.has_vector_observation:
            feat = self.vector_net(*obs.vector)
        elif self.obs_spec.has_visual_observation:
            feat = self.visual_net(*obs.visual)
        else:
            raise Exception("observation must not be empty.")

        if self.use_encoder:
            feat = self.encoder_net(feat)  # [B*T, X]

        if self.use_rnn:
            batch_size = tf.shape(cell_state[0])[0]
            # reshape feature from [B*T, x] to [B, T, x]
            feat = tf.reshape(feat, (batch_size, -1, feat.shape[-1]))
            feat, cell_state = self.memory_net(feat, *cell_state)
            # reshape feature from [B, T, x] to [B*T, x]
            feat = tf.reshape(feat, (-1, tf.shape(feat)[-1]))

        return feat, cell_state

    @property
    def trainable_variables(self):
        tv = []
        tv += self.vector_net.trainable_variables
        tv += self.visual_net.trainable_variables
        if self.use_encoder:
            tv += self.encoder_net.trainable_variables
        if self.use_rnn:
            tv += self.memory_net.trainable_variables
        return tv

    @property
    def weights(self):
        ws = []
        ws += self.vector_net.weights
        ws += self.visual_net.weights
        if self.use_encoder:
            ws += self.encoder_net.weights
        if self.use_rnn:
            ws += self.memory_net.weights
        return ws

    @property
    def _policy_models(self):
        models = {}
        models.update({self.name + '/' + 'vector_net': self.vector_net})
        models.update({self.name + '/' + 'visual_net': self.visual_net})
        if self.use_encoder:
            models.update({self.name + '/' + 'encoder_net': self.encoder_net})
        if self.use_rnn:
            models.update({self.name + '/' + 'memory_net': self.memory_net})
        return models

    @property
    def _all_models(self):
        models = {}
        models.update({self.name + '/' + 'vector_net': self.vector_net})
        models.update({self.name + '/' + 'visual_net': self.visual_net})
        if self.use_encoder:
            models.update({self.name + '/' + 'encoder_net': self.encoder_net})
        if self.use_rnn:
            models.update({self.name + '/' + 'memory_net': self.memory_net})
        return models

    def _copy(self, name='_representation_target_net'):
        copy_net = self.__class__(name=name,
                                  obs_spec=self.obs_spec,
                                  representation_net_params=self.representation_net_params)
        update_target_net_weights(copy_net.weights, self.weights)
        return copy_net


class MultiAgentCentralCriticRepresentationNetwork(RepresentationNetwork):
    '''
      visual -> visual_net -> feat ↘
                                     feat -> encoder_net -> feat ↘                ↗ feat
      vector -> vector_net -> feat ↗                             -> memory_net ->
                                                      cell_state ↗                ↘ cell_state
    '''

    def __init__(self,
                 name: str,
                 obs_spec_list: List[ObsSpec],
                 representation_net_params: Dict):
        super().__init__(name)

        self.obs_spec_list = obs_spec_list
        self.representation_nets = []
        for i, obs_spec in enumerate(self.obs_spec_list):
            self.representation_nets.append(
                DefaultRepresentationNetwork(name=name+f'_{i}',
                                             obs_spec=obs_spec,
                                             representation_net_params=representation_net_params)
            )
        self.h_dim = sum([rep_net.h_dim for rep_net in self.representation_nets])

    @tf.function
    def __call__(self, obss, cell_state):
        # TODO: cell_state
        output = []
        for obs, rep_net in zip(obss, self.representation_nets):
            output.append(rep_net(obs, cell_state=cell_state)[0])
        feats = tf.concat(output, axis=-1)
        return feats, cell_state

    @property
    def trainable_variables(self):
        tv = []
        for rep_net in self.representation_nets:
            tv += rep_net.trainable_variables
        return tv

    @property
    def weights(self):
        ws = []
        for rep_net in self.representation_nets:
            ws += rep_net.weights
        return ws

    @property
    def _policy_models(self):
        models = {}
        for rep_net in self.representation_nets:
            models.update(rep_net._policy_models)
        return models

    @property
    def _all_models(self):
        models = {}
        for rep_net in self.representation_nets:
            models.update(rep_net._all_models)
        return models

    def _copy(self, name='_representation_target_net'):
        copy_net = self.__class__(name=name,
                                  obs_spec_list=self.obs_spec_list,
                                  representation_net_params=self.representation_net_params)
        update_target_net_weights(copy_net.weights, self.weights)
        return copy_net


class ValueNetwork:
    '''
    feat -> value_net -> outputs
    '''

    def __init__(self,
                 name: str = 'test',
                 representation_net: RepresentationNetwork = None,
                 train_representation_net: bool = True,

                 value_net_type: OutputNetworkType = None,
                 value_net_kwargs: dict = {}):
        assert value_net_type is not None, 'assert value_net_type is not None'
        super().__init__()
        self.name = name
        self.representation_net = representation_net
        self.train_representation_net = train_representation_net
        if self.representation_net is not None:
            value_net_kwargs.update(dict(vector_dim=self.representation_net.h_dim))
        self.value_net = get_output_network_from_type(value_net_type)(**value_net_kwargs)

    def __call__(self, obs, *args, cell_state=(None,), **kwargs):
        # feature [B, x]
        assert self.representation_net is not None, 'self.representation_net is not None'
        ret = {}
        ret['feat'], ret['cell_state'] = self.get_feat(obs, cell_state, out_cell_state=True)
        ret['value'] = self.value_net(ret['feat'], *args, **kwargs)
        return ret

    def get_feat(self, obs, cell_state, out_cell_state=False):
        feat, cell_state = self.representation_net(obs, cell_state)
        if out_cell_state:
            return feat, cell_state
        else:
            return feat

    def get_value(self, feat, *args, **kwargs):
        output = self.value_net(feat, *args, **kwargs)
        return output

    @property
    def trainable_variables(self):
        tv = []
        if self.representation_net and self.train_representation_net:
            tv += self.representation_net.trainable_variables
        tv += self.value_net.trainable_variables
        return tv

    @property
    def weights(self):
        ws = []
        if self.representation_net:
            ws += self.representation_net.weights
        ws += self.value_net.weights
        return ws

    @property
    def _policy_models(self):
        models = {}
        if self.representation_net:
            models.update(self.representation_net._policy_models)
        models.update({self.name + '/' + 'value_net': self.value_net})
        return models

    @property
    def _all_models(self):
        models = {}
        if self.representation_net and self.train_representation_net:
            models.update(self.representation_net._all_models)
        models.update({self.name + '/' + 'value_net': self.value_net})
        return models


class DoubleValueNetwork(ValueNetwork):
    '''
         ↗ value_net1 -> outputs
    feat
         ↘ value_net2 -> outputs
    '''

    def __init__(self,
                 name: str = 'test',
                 representation_net: RepresentationNetwork = None,
                 train_representation_net: bool = True,

                 value_net_type: OutputNetworkType = None,
                 value_net_kwargs: dict = {}):
        super().__init__(name, representation_net, train_representation_net, value_net_type, value_net_kwargs)
        if self.representation_net is not None:
            value_net_kwargs.update(dict(vector_dim=self.representation_net.h_dim))
        self.value_net2 = get_output_network_from_type(value_net_type)(**value_net_kwargs)

    def __call__(self, obs, *args, cell_state=(None,), **kwargs):
        # feature [B, x]
        assert self.representation_net is not None, 'self.representation_net is not None'
        ret = {}
        ret['feat'], ret['cell_state'] = self.get_feat(obs, cell_state, out_cell_state=True)
        ret['value'] = self.value_net(ret['feat'], *args, **kwargs)
        ret['value2'] = self.value_net2(ret['feat'], *args, **kwargs)
        return ret

    def get_value(self, feat, *args, **kwargs):
        output = self.value_net(feat, *args, **kwargs)
        output2 = self.value_net2(feat, *args, **kwargs)
        return output, output2

    def get_min(self, *args, **kwargs):
        return tf.minimum(*self.get_value(*args, **kwargs))

    @property
    def trainable_variables(self):
        return super().trainable_variables + self.value_net2.trainable_variables

    @property
    def weights(self):
        return super().weights + self.value_net2.weights

    @property
    def _all_models(self):
        models = super()._all_models
        models.update({self.name + '/' + 'value_net2': self.value_net2})
        return models


class ACNetwork(ValueNetwork):
    '''
         ↗ policy_net -> outputs
    feat
         ↘ value_net  -> outputs
    '''

    def __init__(self,
                 name: str = 'test',
                 representation_net: RepresentationNetwork = None,
                 train_representation_net: bool = True,

                 policy_net_type: OutputNetworkType = None,
                 policy_net_kwargs: dict = {},

                 value_net_type: OutputNetworkType = None,
                 value_net_kwargs: dict = {}):

        super().__init__(name, representation_net, train_representation_net, value_net_type, value_net_kwargs)
        if self.representation_net is not None:
            policy_net_kwargs.update(dict(vector_dim=self.representation_net.h_dim))
        self.policy_net = get_output_network_from_type(policy_net_type)(**policy_net_kwargs)

    def __call__(self, obs, *args, cell_state=(None,), **kwargs):
        # feature [B, x]
        assert self.representation_net is not None, 'self.representation_net is not None'
        ret = {}
        ret['feat'], ret['cell_state'] = self.get_feat(obs, cell_state, out_cell_state=True)
        ret['actor'] = self.policy_net(ret['feat'])
        if args or kwargs:
            ret['critic'] = self.value_net(ret['feat'], *args, **kwargs)
        else:
            ret['critic'] = None
        return ret

    @property
    def actor_trainable_variables(self):
        return self.policy_net.trainable_variables

    @property
    def critic_trainable_variables(self):
        return super().trainable_variables

    @property
    def weights(self):
        return super().weights + self.policy_net.weights

    @property
    def _policy_models(self):
        '''重载'''
        models = super()._policy_models
        models.update({self.name + '/' + 'policy_net': self.policy_net})
        return models

    @property
    def _all_models(self):
        models = super()._all_models
        models.update({self.name + '/' + 'policy_net': self.policy_net})
        return models


class ACCNetwork(ACNetwork):
    '''
    Use for PD-DDPG

         ↗ policy_net -> outputs
    feat -> value_net  -> outputs
         ↘ value_net2  -> outputs
    '''

    def __init__(self,
                 name: str = 'test',
                 representation_net: RepresentationNetwork = None,
                 train_representation_net: bool = True,

                 policy_net_type: OutputNetworkType = None,
                 policy_net_kwargs: dict = {},

                 value_net_type: OutputNetworkType = None,
                 value_net_kwargs: dict = {},

                 value_net2_type: OutputNetworkType = None,
                 value_net2_kwargs: dict = {}):

        super().__init__(name, representation_net, train_representation_net,
                         policy_net_type, policy_net_kwargs,
                         value_net_type, value_net_kwargs)
        if self.representation_net is not None:
            value_net2_kwargs.update(dict(vector_dim=self.representation_net.h_dim))
        self.value_net2 = get_output_network_from_type(value_net2_type)(**value_net2_kwargs)

    def __call__(self, obs, *args, cell_state=(None,), **kwargs):
        # feature [B, x]
        assert self.representation_net is not None, 'self.representation_net is not None'
        ret = {}
        ret['feat'], ret['cell_state'] = self.get_feat(obs, cell_state, out_cell_state=True)
        ret['actor'] = self.policy_net(ret['feat'])
        if args or kwargs:
            ret['critic'] = self.value_net(ret['feat'], *args, **kwargs)
            ret['critic2'] = self.value_net2(ret['feat'], *args, **kwargs)
        else:
            ret['critic'] = None
            ret['critic2'] = None
        return ret

    @property
    def critic_trainable_variables(self):
        return super().critic_trainable_variables + self.value_net2.trainable_variables

    @property
    def value_net_trainable_variables(self):
        return super().critic_trainable_variables

    @property
    def value_net2_trainable_variables(self):
        return self.value_net2.trainable_variables

    @property
    def weights(self):
        return super().weights + self.value_net2.weights

    @property
    def _all_models(self):
        models = super()._all_models
        models.update({self.name + '/' + 'value_net2': self.value_net2})
        return models


class ADoubleCNetwork(ACNetwork):
    '''

         ↗ policy_net -> outputs
    feat -> value_net  -> outputs
         ↘ value_net2  -> outputs
    '''

    def __init__(self,
                 name: str = 'test',
                 representation_net: RepresentationNetwork = None,
                 train_representation_net: bool = True,

                 policy_net_type: OutputNetworkType = None,
                 policy_net_kwargs: dict = {},

                 value_net_type: OutputNetworkType = None,
                 value_net_kwargs: dict = {}):
        super().__init__(name, representation_net, train_representation_net,
                         policy_net_type, policy_net_kwargs,
                         value_net_type, value_net_kwargs)
        if self.representation_net is not None:
            value_net_kwargs.update(dict(vector_dim=self.representation_net.h_dim))
        self.value_net2 = get_output_network_from_type(value_net_type)(**value_net_kwargs)

    def __call__(self, obs, *args, cell_state=(None,), **kwargs):
        # feature [B, x]
        assert self.representation_net is not None, 'self.representation_net is not None'
        ret = {}
        ret['feat'], ret['cell_state'] = self.get_feat(obs, cell_state, out_cell_state=True)
        ret['actor'] = self.policy_net(feat)
        if args or kwargs:
            ret['critic'] = self.value_net(ret['feat'], *args, **kwargs)
            ret['critic2'] = self.value_net2(ret['feat'], *args, **kwargs)
        else:
            ret['critic'] = None
            ret['critic2'] = None
        return ret

    def get_value(self, feat, *args, **kwargs):
        output = self.value_net(feat, *args, **kwargs)
        output2 = self.value_net2(feat, *args, **kwargs)
        return output, output2

    def get_min(self, *args, **kwargs):
        return tf.minimum(*self.get_value(*args, **kwargs))

    @property
    def critic_trainable_variables(self):
        return super().trainable_variables + self.value_net2.trainable_variables

    @property
    def weights(self):
        return super().weights + self.value_net2.weights

    @property
    def _all_models(self):
        models = super()._all_models
        models.update({self.name + '/' + 'value_net2': self.value_net2})
        return models
