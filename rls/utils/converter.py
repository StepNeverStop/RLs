
import torch as t
import numpy as np

from numbers import Number

from rls.common.specs import Data
from rls.utils.display import colorize
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


def to_numpy(x):
    try:
        if isinstance(x, t.Tensor):  # tensor -> numpy
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
            raise Exception(f'Data: {x}.\n Unexpected data type when convert data to Numpy: {type(x)}')
    except Exception as e:
        logger.error(x)
        logger.error(e)


def to_tensor(x, dtype=t.float32, device='cpu'):
    if x is None or isinstance(x, Number):
        return x
    try:
        if isinstance(x, Data):
            return x.convert(func=lambda y: t.as_tensor(y, dtype=dtype, device=device))
        elif isinstance(x, np.ndarray):
            return t.from_numpy(x).type(dtype).to(device)
        elif isinstance(x, t.Tensor):
            return x.type(dtype).to(device)
        elif isinstance(x, dict):
            return {k: to_tensor(v, dtype=dtype, device=device) for k, v in x.items()}
        elif isinstance(x, (tuple, list)):
            return [to_tensor(_x, dtype=dtype, device=device) for _x in x]
        else:
            raise Exception(f'Data: {x}.\n Unexpected data type when convert data to Torch.Tensor: {type(x)}')
    except Exception as e:
        logger.error(x)
        logger.error(e)
