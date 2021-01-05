

import numpy as np
import tensorflow as tf

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
from rls.utils.specs import ObsSpec
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


class DefaultRepresentationNetwork(RepresentationNetwork):
    '''
      visual -> visual_net -> feat ↘
                                     feat -> encoder_net -> feat ↘                ↗ feat
      vector -> vector_net -> feat ↗                             -> memory_net ->
                                                      cell_state ↗                ↘ cell_state
    '''

    def __init__(self,
                 obs_spec: ObsSpec,
                 name: str = 'test',
                 vector_net_kwargs: dict = {},
                 visual_net_kwargs: dict = {},
                 encoder_net_kwargs: dict = {},
                 memory_net_kwargs: dict = {}):
        super().__init__(name)

        self.obs_spec = obs_spec

        self.vector_net = MultiVectorNetwork(obs_spec.vector_dims, **vector_net_kwargs)
        logger.debug('initialize vector network successfully.')
        self.visual_net = MultiVisualNetwork(obs_spec.visual_dims, **visual_net_kwargs)
        logger.debug('initialize visual network successfully.')

        encoder_dim = self.vector_net.h_dim + self.visual_net.h_dim
        self.encoder_net = EncoderNetwork(encoder_dim, **encoder_net_kwargs)
        logger.debug('initialize encoder network successfully.')

        memory_dim = self.encoder_net.h_dim
        self.memory_net = MemoryNetwork(memory_dim, **memory_net_kwargs)
        logger.debug('initialize memory network successfully.')

        self.h_dim = self.memory_net.h_dim

    @tf.function
    def __call__(self, obs, cell_state, *, need_split=False):
        '''
        params:
            cell_state: Tuple([B, z],)
        return:
            feat: [B, a]
            cell_state: Tuple([B, z],)
        '''
        feat = self.get_encoder_feature(obs)    # [B*T, X]

        if self.memory_net.use_rnn:
            batch_size = tf.shape(cell_state[0])[0]
            # reshape feature from [B*T, x] to [B, T, x]
            feat = tf.reshape(feat, (batch_size, -1, feat.shape[-1]))
            feat, cell_state = self.memory_net(feat, *cell_state)

            if need_split:
                # reshape feature from [B, T+1, x] to ([B*T, x], [B*T, x])
                feat = (tf.reshape(feat[:, :-1], [-1, tf.shape(feat)[-1]]),
                        tf.reshape(feat[:, 1:], [-1, tf.shape(feat)[-1]]))
            else:
                # reshape feature from [B, T, x] to [B*T, x]
                feat = tf.reshape(feat, (-1, tf.shape(feat)[-1]))
        else:
            if need_split:
                feat = tf.split(feat, num_or_size_splits=2, axis=0)

        return feat, cell_state

    def get_encoder_feature(self, obs):
        '''
        params:
            obs: xxx
        return:
            feat: [B, z]
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

        return self.encoder_net(feat)

    @property
    def trainable_variables(self):
        tv = []
        tv += self.vector_net.trainable_variables
        tv += self.visual_net.trainable_variables
        tv += self.encoder_net.trainable_variables
        tv += self.memory_net.trainable_variables
        return tv

    @property
    def weights(self):
        ws = []
        ws += self.vector_net.weights
        ws += self.visual_net.weights
        ws += self.encoder_net.weights
        ws += self.memory_net.weights
        return ws

    @property
    def _policy_models(self):
        models = {}
        models.update({self.name + '/' + 'vector_net': self.vector_net})
        models.update({self.name + '/' + 'visual_net': self.visual_net})
        models.update({self.name + '/' + 'encoder_net': self.encoder_net})
        models.update({self.name + '/' + 'memory_net': self.memory_net})
        return models

    @property
    def _all_models(self):
        models = {}
        models.update({self.name + '/' + 'vector_net': self.vector_net})
        models.update({self.name + '/' + 'visual_net': self.visual_net})
        models.update({self.name + '/' + 'encoder_net': self.encoder_net})
        models.update({self.name + '/' + 'memory_net': self.memory_net})
        return models


class ValueNetwork:
    '''
    feat -> value_net -> outputs
    '''

    def __init__(self,
                 name: str = 'test',
                 representation_net: RepresentationNetwork = None,

                 value_net_type: OutputNetworkType = None,
                 value_net_kwargs: dict = {}):
        assert value_net_type is not None, 'assert value_net_type is not None'
        super().__init__()
        self.name = name
        self.representation_net = representation_net
        if self.representation_net is not None:
            self.value_net = get_output_network_from_type(value_net_type)(
                vector_dim=self.representation_net.h_dim, **value_net_kwargs)
        else:
            self.value_net = get_output_network_from_type(value_net_type)(
                **value_net_kwargs)

    def __call__(self, obs, *args, cell_state=(None,), **kwargs):
        # feature [B, x]
        assert self.representation_net is not None, 'self.representation_net is not None'
        feat, cell_state = self.representation_net(obs, cell_state)
        output = self.value_net(feat, *args, **kwargs)
        return output, cell_state

    def get_value(self, feat, *args, **kwargs):
        output = self.value_net(feat, *args, **kwargs)
        return output

    @property
    def trainable_variables(self):
        tv = self.representation_net.trainable_variables if self.representation_net else []
        tv += self.value_net.trainable_variables
        return tv

    @property
    def weights(self):
        ws = self.representation_net.weights if self.representation_net else []
        ws += self.value_net.weights
        return ws

    @property
    def _policy_models(self):
        models = self.representation_net._policy_models if self.representation_net else {}
        models.update({self.name + '/' + 'value_net': self.value_net})
        return models

    @property
    def _all_models(self):
        models = self.representation_net._all_models if self.representation_net else {}
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

                 value_net_type: OutputNetworkType = None,
                 value_net_kwargs: dict = {}):
        super().__init__(name, representation_net, value_net_type, value_net_kwargs)
        if self.representation_net is not None:
            self.value_net2 = get_output_network_from_type(value_net_type)(
                vector_dim=self.representation_net.h_dim, **value_net_kwargs)
        else:
            self.value_net2 = get_output_network_from_type(value_net_type)(
                **value_net_kwargs)

    def __call__(self, obs, *args, cell_state=(None,), **kwargs):
        # feature [B, x]
        feat, cell_state = self.representation_net(obs, cell_state)
        output = self.value_net(feat, *args, **kwargs)
        output2 = self.value_net2(feat, *args, **kwargs)
        return output, output2, cell_state

    def get_value(self, feat, *args, **kwargs):
        output = self.value_net(feat, *args, **kwargs)
        output2 = self.value_net2(feat, *args, **kwargs)
        return output, output2

    def get_min(self, *args, **kwargs):
        return tf.minimum(*self.get_value(*args, **kwargs))

    def get_max(self, *args, **kwargs):
        return tf.maximum(*self.get_value(*args, **kwargs))

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

                 policy_net_type: OutputNetworkType = None,
                 policy_net_kwargs: dict = {},

                 value_net_type: OutputNetworkType = None,
                 value_net_kwargs: dict = {}):

        super().__init__(name, representation_net, value_net_type, value_net_kwargs)
        if self.representation_net is not None:
            self.policy_net = get_output_network_from_type(policy_net_type)(
                vector_dim=self.representation_net.h_dim, **policy_net_kwargs)
        else:
            self.policy_net = get_output_network_from_type(policy_net_type)(
                **policy_net_kwargs)

    def __call__(self, obs, *args, cell_state=(None,), **kwargs):
        # feature [B, x]
        feat, cell_state = self.representation_net(obs, cell_state)
        output = self.policy_net(feat, *args, **kwargs)
        return output, cell_state

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

                 policy_net_type: OutputNetworkType = None,
                 policy_net_kwargs: dict = {},

                 value_net_type: OutputNetworkType = None,
                 value_net_kwargs: dict = {},

                 value_net2_type: OutputNetworkType = None,
                 value_net2_kwargs: dict = {}):

        super().__init__(name, representation_net,
                         policy_net_type, policy_net_kwargs,
                         value_net_type, value_net_kwargs)
        if self.representation_net is not None:
            self.value_net2 = get_output_network_from_type(value_net2_type)(
                vector_dim=self.representation_net.h_dim, **value_net2_kwargs)
        else:
            self.value_net2 = get_output_network_from_type(value_net2_type)(
                **value_net2_kwargs)

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

                 policy_net_type: OutputNetworkType = None,
                 policy_net_kwargs: dict = {},

                 value_net_type: OutputNetworkType = None,
                 value_net_kwargs: dict = {}):
        super().__init__(name, representation_net,
                         policy_net_type, policy_net_kwargs,
                         value_net_type, value_net_kwargs)
        if self.representation_net is not None:
            self.value_net2 = get_output_network_from_type(value_net_type)(
                vector_dim=self.representation_net.h_dim, **value_net_kwargs)
        else:
            self.value_net2 = get_output_network_from_type(value_net_type)(
                **value_net_kwargs)

    def get_value(self, feat, *args, **kwargs):
        output = self.value_net(feat, *args, **kwargs)
        output2 = self.value_net2(feat, *args, **kwargs)
        return output, output2

    def get_min(self, *args, **kwargs):
        return tf.minimum(*self.get_value(*args, **kwargs))

    def get_max(self, *args, **kwargs):
        return tf.maximum(*self.get_value(*args, **kwargs))

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
