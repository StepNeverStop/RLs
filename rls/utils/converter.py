from numbers import Number

import numpy as np
import torch as th

from rls.common.data import Data
from rls.utils.logging_utils import get_logger

logger = get_logger(__name__)


def to_numpy(x):
    try:
        if isinstance(x, Data):
            return x.convert(func=lambda y: to_numpy(y))
        elif isinstance(x, th.Tensor):  # tensor -> numpy
            return x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray) or x is None:  # second often case
            return x
        elif isinstance(x, (np.number, np.bool_, Number)):
            return np.asanyarray(x)
        elif isinstance(x, dict):
            return {k: to_numpy(v) for k, v in x.items()}
        elif isinstance(x, (tuple, list)):
            return [to_numpy(_x) for _x in x]
        else:
            raise Exception(
                f'Data: {x}.\n Unexpected data type when convert data to Numpy: {type(x)}')
    except Exception as e:
        logger.error(x)
        logger.error(e)


# noinspection PyTypeChecker
def to_tensor(x, dtype=th.float32, device: str = 'cpu'):
    if x is None or isinstance(x, Number):
        return x
    try:
        if isinstance(x, Data):
            return x.convert(func=lambda y: th.as_tensor(y, dtype=dtype, device=device))
        elif isinstance(x, np.ndarray):
            return th.from_numpy(x).type(dtype).to(device)
        elif isinstance(x, th.Tensor):
            return x.type(dtype).to(device)
        elif isinstance(x, dict):
            return {k: to_tensor(v, dtype=dtype, device=device) for k, v in x.items()}
        elif isinstance(x, (tuple, list)):
            return [to_tensor(_x, dtype=dtype, device=device) for _x in x]
        else:
            raise Exception(
                f'Data: {x}.\n Unexpected data type when convert data to Torch.Tensor: {type(x)}')
    except Exception as e:
        logger.error(x)
        logger.error(e)
