from numbers import Number

import numpy as np
import torch as th

from rls.common.data import Data
from rls.utils.logging_utils import get_logger

logger = get_logger(__name__)


def to_numpy_or_number(x):
    try:
        if isinstance(x, Data):
            return x.convert(func=lambda y: to_numpy_or_number(y))
        elif isinstance(x, th.Tensor):  # tensor -> numpy
            x = x.detach().cpu()
            x = x.item() if x.ndim == 0 else x.numpy()
            return x
        elif isinstance(x, np.ndarray):  # second often case
            return x.item() if x.ndim == 0 else x
        elif x is None or isinstance(x, Number):
            return x
        elif isinstance(x, dict):
            return {k: to_numpy_or_number(v) for k, v in x.items()}
        elif isinstance(x, (tuple, list)):
            return [to_numpy_or_number(_x) for _x in x]
        elif isinstance(x, (np.number, np.bool_)):
            return np.asanyarray(x)
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
