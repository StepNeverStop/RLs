

import torch as t

from typing import (List,
                    Dict)


from rls.nn.networks import (MultiVectorNetwork,
                             MultiVisualNetwork,
                             EncoderNetwork,
                             MemoryNetwork)
from rls.utils.logging_utils import get_logger
from rls.common.specs import ObsSpec
logger = get_logger(__name__)

Rep_REGISTER = {}


class RepresentationNetwork(t.nn.Module):

    def __init__(self):
        super().__init__()
        self.h_dim = None

    def forward(self):
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
                 representation_net_params: Dict):
        super().__init__()

        self.obs_spec = obs_spec
        self.representation_net_params = representation_net_params

        vector_net_params = dict(representation_net_params.get('vector_net_params', {}))
        visual_net_params = dict(representation_net_params.get('visual_net_params', {}))

        logger.debug('initialize vector network begin.')
        self.vector_net = MultiVectorNetwork(obs_spec.vector_dims, **vector_net_params)
        logger.debug('initialize vector network successfully.')

        logger.debug('initialize visual network begin.')
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
            self.memory_net = MemoryNetwork(self.h_dim, **memory_net_params)
            logger.debug('initialize memory network successfully.')
            self.h_dim = self.memory_net.h_dim

    def forward(self, obs, cell_state):
        '''
        params:
            cell_state: Tuple([B, z],)
        return:
            feat: [B, a]
            cell_state: Tuple([B, z],)
        '''
        if self.obs_spec.has_vector_observation and self.obs_spec.has_visual_observation:
            vec_feat = self.vector_net(*obs.vector.__dict__.values())
            vis_feat = self.visual_net(*obs.visual.__dict__.values())
            feat = t.cat([vec_feat, vis_feat], -1)
        elif self.obs_spec.has_vector_observation:
            feat = self.vector_net(*obs.vector.__dict__.values())
        elif self.obs_spec.has_visual_observation:
            feat = self.visual_net(*obs.visual.__dict__.values())
        else:
            raise Exception("observation must not be empty.")

        if self.use_encoder:
            feat = self.encoder_net(feat)  # [B*T, X]

        if self.use_rnn:
            batch_size = cell_state[0].shape[0]
            # reshape feature from [B*T, x] to [B, T, x]

            feat = feat.view(batch_size, -1, feat.shape[-1])
            feat, cell_state = self.memory_net(feat, *cell_state)
            # reshape feature from [B, T, x] to [B*T, x]
            feat = feat.view(-1, feat.shape[-1])

        return feat, cell_state


class MultiAgentCentralCriticRepresentationNetwork(RepresentationNetwork):
    '''
      visual -> visual_net -> feat ↘
                                     feat -> encoder_net -> feat ↘                ↗ feat
      vector -> vector_net -> feat ↗                             -> memory_net ->
                                                      cell_state ↗                ↘ cell_state
    '''

    def __init__(self,
                 obs_spec_list: List[ObsSpec],
                 representation_net_params: Dict):
        super().__init__()

        self.obs_spec_list = obs_spec_list
        self.representation_nets = []
        for i, obs_spec in enumerate(self.obs_spec_list):
            self.representation_nets.append(
                DefaultRepresentationNetwork(obs_spec=obs_spec,
                                             representation_net_params=representation_net_params)
            )
        self.h_dim = sum([rep_net.h_dim for rep_net in self.representation_nets])

    def forward(self, obss, cell_state):
        # TODO: cell_state
        output = []
        for obs, rep_net in zip(obss, self.representation_nets):
            output.append(rep_net(obs, cell_state=cell_state)[0])
        feats = t.cat(output, -1)
        return feats, cell_state


Rep_REGISTER['default'] = DefaultRepresentationNetwork
Rep_REGISTER['multi'] = MultiAgentCentralCriticRepresentationNetwork
