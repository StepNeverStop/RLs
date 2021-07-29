
import torch as t

from copy import deepcopy
from typing import (Dict,
                    Callable,
                    Union,
                    List,
                    NoReturn,
                    Optional,
                    Any)

from rls.utils.display import colorize
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class IndependentMA:

    def __init__(self, sarl_model_class, envspecs, algo_args):

        self._n_agents = len(envspecs)
        if self._n_agents > 1:
            logger.info(colorize('using SARL algorithm to train Multi-Agent task, model has been changed to independent-SARL automatically.'))

        self.models = []
        for i in range(self._n_agents):
            _algo_args = deepcopy(algo_args)
            if self._n_agents > 1:
                _algo_args.base_dir += f'/i{sarl_model_class.__name__}-{i}'
            self.models.append(sarl_model_class(envspec=envspecs[i], **_algo_args))

    def __call__(self, obs, evaluation=False):
        actions = []
        for model, _obs in zip(self.models, obs):
            actions.append(model(_obs, evaluation))
        return actions

    def reset(self):
        for model in self.models:
            model.reset()

    def prefill_store(self, data):    # TODO: remove
        for model, d in zip(self.models, data):
            model.store_data(d)

    def store_data(self, data):
        for model, d in zip(self.models, data):
            model.store_data(d)

    def partial_reset(self, dones):
        for model, d in zip(self.models, dones):
            model.partial_reset(d)

    def learn(self, **kwargs):
        for model in self.models:
            model.learn(**kwargs)

    def save(self, **kwargs):
        for model in self.models:
            model.save(**kwargs)

    def resume(self, base_dir: Optional[str] = None) -> Dict:
        for i, model in enumerate(self.models):
            if self._n_agents > 1:
                base_dir += f'/i{model.__class__.__name__}-{i}'
            training_info = model.resume(base_dir)
        else:
            return training_info

    def write_summaries(self,
                        global_step: Union[int, t.Tensor],
                        summaries: Dict,
                        writer=None) -> NoReturn:
        '''
        write summaries showing in tensorboard.
        '''
        for i, summary in summaries.items():
            self.models[i].write_summaries(global_step, summaries=summary)

    def close(self):
        for model in self.models:
            model.close()
