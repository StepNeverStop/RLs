
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

    def __init__(self, sarl_model_class, algo_args, n):

        logger.info(colorize('using SARL algorithm to train Multi-Agent task, model has been changed to independent-SARL automatically.'))

        self.models = []
        for i in range(n):
            _algo_args = deepcopy(algo_args)
            _algo_args.base_dir += f'/i{sarl_model_class.__name__}-{i}'
            self.models.append(sarl_model_class(**_algo_args))

    def __call__(self, obss, evaluation=False):
        actions = []
        for model, obs in zip(self.models, obss):
            actions.append(model(obs, evaluation))
        return actions

    def reset(self):
        for model in self.models:
            model.reset()

    def store_data(self, data):
        for model, d in zip(self.modes, data):
            model.store_data(d)

    def partial_reset(self, dones):
        for model, d in zip(self.modes, dones):
            model.partial_reset(d)

    def learn(self, **kwargs):
        for model in self.models:
            model.learn(**kwargs)

    def save(self, **kwargs):
        for model in self.models:
            model.save(**kwargs)

    def resume(self, base_dir: Optional[str] = None) -> NoReturn:
        for i, model in enumerate(self.models):
            model.resume(base_dir+f'/i{model.__class__.__name__}-{i}')

    def get_init_training_info(self) -> Dict:
        return self.models[0].get_init_training_info()

    def write_summaries(self,
                        global_step: Union[int, t.Tensor],
                        summaries: Dict,
                        writer=None) -> NoReturn:
        '''
        write tf summaries showing in tensorboard.
        '''
        for i, summary in summaries.items():
            self.models[i].write_summaries(global_step, summaries=summary)

    def close(self):
        for model in self.models:
            model.close()
