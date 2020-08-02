import grpc
import time

from concurrent import futures

from rls.distribute.pb2 import \
    apex_datatype_pb2, \
    apex_learner_pb2_grpc
from rls.distribute.utils.apex_utils import \
    proto2numpy, \
    batch_proto2numpy, \
    batch_numpy2proto
from rls.distribute.utils.check import check_port_in_use


class LearnerServicer(apex_learner_pb2_grpc.LearnerServicer):

    def __init__(self, model):
        self.model = model

    def SendNumpyArray(self, request: apex_datatype_pb2.NDarray, context) -> apex_datatype_pb2.Nothing:
        arr = proto2numpy(request)
        print(arr)
        return apex_datatype_pb2.Nothing()

    def SendBatchNumpyArray(self, request, context) -> apex_datatype_pb2.Nothing:
        arr_list = batch_proto2numpy(request)
        print(arr_list)
        return apex_datatype_pb2.Nothing()

    def GetParams(self, request: apex_datatype_pb2.Nothing, context) -> apex_datatype_pb2.ListNDarray:
        return batch_numpy2proto(self.model.get_worker_params())


def learner(ip, port, model):
    for i in range(10):
        if check_port_in_use(port, ip):
            print(f'{i}: port {port} is under used.')
            time.sleep(1)
        else:
            break
    else:
        raise Exception('Cannot start learner correctly.')

    server = grpc.server(futures.ThreadPoolExecutor())
    apex_learner_pb2_grpc.add_LearnerServicer_to_server(LearnerServicer(model), server)
    server.add_insecure_port(':'.join([ip, port]))
    server.start()
    print('start learner success.')
    server.wait_for_termination()
