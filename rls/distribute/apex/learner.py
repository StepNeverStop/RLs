import grpc
import time

from concurrent import futures

from rls.distribute.pb2 import \
    apex_datatype_pb2, \
    apex_learner_pb2_grpc
from rls.distribute.utils.apex_utils import \
    numpy2proto, \
    proto2numpy, \
    batch_proto2numpy, \
    batch_numpy2proto, \
    proto2exps_and_prios
from rls.distribute.utils.check import check_port_in_use
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class LearnerServicer(apex_learner_pb2_grpc.LearnerServicer):

    def __init__(self, model):
        self.model = model

    def SendNumpyArray(self, request: apex_datatype_pb2.NDarray, context) -> apex_datatype_pb2.Nothing:
        arr = proto2numpy(request)
        print(arr)
        return apex_datatype_pb2.Nothing()

    def SendBatchNumpyArray(self, request: apex_datatype_pb2.ListNDarray, context) -> apex_datatype_pb2.Nothing:
        arr_list = batch_proto2numpy(request)
        print(arr_list)
        return apex_datatype_pb2.Nothing()

    def GetParams(self, request: apex_datatype_pb2.Nothing, context) -> apex_datatype_pb2.ListNDarray:
        params = batch_numpy2proto(self.model.get_worker_params())
        logger.info('send params to worker.')
        return params

    def SendExperienceGetPriorities(self, request: apex_datatype_pb2.ExpsAndPrios, context) -> apex_datatype_pb2.NDarray:
        data, prios = proto2exps_and_prios(request)
        td_error = numpy2proto(self.model.apex_learn(data, prios))
        logger.info('send new priorities to buffer.')
        return td_error


def learner(ip, port, model):
    for i in range(10):
        if check_port_in_use(port, ip):
            print(f'{i}: port {port} is under used.')
            time.sleep(1)
        else:
            break
    else:
        raise Exception('Cannot start learner correctly.')

    assert hasattr(model, 'apex_learn'), 'this algorithm does not support Ape-X learning for now.'

    server = grpc.server(futures.ThreadPoolExecutor())
    apex_learner_pb2_grpc.add_LearnerServicer_to_server(LearnerServicer(model), server)
    server.add_insecure_port(':'.join([ip, port]))
    server.start()
    print('start learner success.')
    server.wait_for_termination()
