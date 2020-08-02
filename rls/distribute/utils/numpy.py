import numpy as np

from io import BytesIO
from typing import List


def bytes2numpy(data: bytes) -> np.ndarray:
    '''
    TODO: Annotation
    '''
    nda_bytes = BytesIO(data)
    nda = np.load(nda_bytes, allow_pickle=False)
    return nda


def numpy2bytes(data: np.ndarray) -> bytes:
    '''
    TODO: Annotation
    '''
    nda_bytes = BytesIO()
    np.save(nda_bytes, data, allow_pickle=False)
    return nda_bytes.getvalue()


def batch_bytes2numpy(data: List[bytes]) -> List[np.ndarray]:
    '''
    TODO: Annotation
    '''
    return list(map(bytes2numpy, data))


def batch_numpy2bytes(data: List[np.ndarray]) -> List[bytes]:
    '''
    TODO: Annotation
    '''
    return list(map(numpy2bytes, data))
