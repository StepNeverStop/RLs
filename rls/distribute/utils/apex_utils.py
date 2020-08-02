import numpy as np

from typing import List

from rls.distribute.pb2 import apex_datatype_pb2
from rls.distribute.utils.numpy import \
    bytes2numpy, \
    numpy2bytes, \
    batch_bytes2numpy, \
    batch_numpy2bytes


def numpy2proto(arr: np.ndarray) -> apex_datatype_pb2.NDarray:
    '''
    TODO: Annotation
    '''
    return apex_datatype_pb2.NDarray(
        data=numpy2bytes(arr)
    )


def proto2numpy(proto: apex_datatype_pb2.NDarray) -> np.ndarray:
    '''
    TODO: Annotation
    '''
    return bytes2numpy(proto.data)


def batch_numpy2proto(arr_list: List[np.ndarray]) -> apex_datatype_pb2.ListNDarray:
    '''
    TODO: Annotation
    '''
    return apex_datatype_pb2.ListNDarray(
        data=batch_numpy2bytes(arr_list)
    )


def batch_proto2numpy(proto: apex_datatype_pb2.ListNDarray) -> List[np.ndarray]:
    '''
    TODO: Annotation
    '''
    return batch_bytes2numpy(proto.data)


def exps_and_prios2proto(exps: List[np.ndarray], prios: np.ndarray) -> apex_datatype_pb2.ExpsAndPrios:
    return apex_datatype_pb2.ExpsAndPrios(
        data=batch_numpy2bytes(exps),
        prios=numpy2bytes(prios)
    )

def proto2exps_and_prios(proto: apex_datatype_pb2.ExpsAndPrios):
    return (
        batch_bytes2numpy(proto.data),
        bytes2numpy(proto.prios)
    )
