

from typing import Dict, List

import torch as t

from rls.common.specs import SensorSpec
from rls.nn.networks import (EncoderNetwork, MemoryNetwork, MultiVectorNetwork,
                             MultiVisualNetwork)
from rls.utils.logging_utils import get_logger

logger = get_logger(__name__)

Rep_REGISTER = {}


class RepresentationNetwork(t.nn.Module):
    '''
      visual -> visual_net -> feat ↘
                                     feat -> encoder_net -> feat ↘                ↗ feat
      vector -> vector_net -> feat ↗                             -> memory_net ->
                                                      cell_state ↗                ↘ cell_state
    '''

    def __init__(self,
                 obs_spec: SensorSpec,
                 rep_net_params: Dict):
        super().__init__()

        self.obs_spec = obs_spec
        self._rep_net_params = rep_net_params
        self.h_dim = 0

        if self.obs_spec.has_vector_observation:
            vector_net_params = dict(
                rep_net_params.get('vector_net_params', {}))
            logger.debug('initialize vector network begin.')
            self.vector_net = MultiVectorNetwork(
                obs_spec.vector_dims, **vector_net_params)
            logger.debug('initialize vector network successfully.')
            self.h_dim += self.vector_net.h_dim

        if self.obs_spec.has_visual_observation:
            visual_net_params = dict(
                rep_net_params.get('visual_net_params', {}))
            logger.debug('initialize visual network begin.')
            self.visual_net = MultiVisualNetwork(
                obs_spec.visual_dims, **visual_net_params)
            logger.debug('initialize visual network successfully.')
            self.h_dim += self.visual_net.h_dim

        self.use_other_info = False
        if self.obs_spec.has_other_observation:
            self.use_other_info = True
            self.h_dim += self.obs_spec.other_dims

        encoder_net_params = dict(
            rep_net_params.get('encoder_net_params', {}))
        self.encoder_net = EncoderNetwork(self.h_dim, **encoder_net_params)
        logger.debug('initialize encoder network successfully.')
        self.h_dim = self.encoder_net.h_dim

        memory_net_params = dict(
            rep_net_params.get('memory_net_params', {}))
        self.memory_net = MemoryNetwork(self.h_dim, **memory_net_params)
        logger.debug('initialize memory network successfully.')
        self.h_dim = self.memory_net.h_dim

    def forward(self, obs, cell_state=None, begin_mask=None):
        '''
        params:
            obs: [T, B, *] or [B, *]
        return:
            feat: [T, B, *] or [B, *]
        '''
        feat_list = []

        if self.obs_spec.has_vector_observation:
            feat_list.append(self.vector_net(*obs.vector.values()))

        if self.obs_spec.has_visual_observation:
            feat_list.append(self.visual_net(*obs.visual.values()))

        feat = t.cat(feat_list, -1)  # [T, B, *] or [B, *]

        if self.use_other_info:
            feat = t.cat([feat, obs.other], -1)

        feat = self.encoder_net(feat)  # [T, B, *] or [B, *]

        feat, cell_state = self.memory_net(
            feat, cell_state, begin_mask)    # [T, B, *] or [B, *]
        return feat, cell_state


class MultiAgentCentralCriticRepresentationNetwork(RepresentationNetwork):
    '''
      visual -> visual_net -> feat ↘
                                     feat -> encoder_net -> feat ↘                ↗ feat
      vector -> vector_net -> feat ↗                             -> memory_net ->
                                                      cell_state ↗                ↘ cell_state
    '''

    def __init__(self,
                 obs_spec_list: List[SensorSpec],
                 rep_net_params: Dict):
        super().__init__()

        self.obs_spec_list = obs_spec_list
        self.representation_nets = []
        for i, obs_spec in enumerate(self.obs_spec_list):
            self.representation_nets.append(
                RepresentationNetwork(obs_spec=obs_spec,
                                      rep_net_params=rep_net_params)
            )
        self.h_dim = sum(
            [rep_net.h_dim for rep_net in self.representation_nets])

    def forward(self, obss, cell_state):
        # TODO: cell_state
        output = []
        for obs, rep_net in zip(obss, self.representation_nets):
            output.append(rep_net(obs, cell_state=cell_state)[0])
        feats = t.cat(output, -1)
        return feats, cell_state


Rep_REGISTER['default'] = RepresentationNetwork
Rep_REGISTER['multi'] = MultiAgentCentralCriticRepresentationNetwork
