import grpc
import time

from concurrent import futures

from rls.distribute.pb2 import \
    apex_datatype_pb2, \
    apex_buffer_pb2_grpc
from rls.distribute.utils.apex_utils import \
    proto2numpy, \
    batch_proto2numpy, \
    batch_numpy2proto
from rls.distribute.utils.check import check_port_in_use
from rls.memories.replay_buffer import PrioritizedExperienceReplay


class BufferServicer(apex_buffer_pb2_grpc.BufferServicer):

    def __init__(self, buffer_args):
        self.buffer = PrioritizedExperienceReplay(**buffer_args)

    def SendTrajectories(self, request_iterator, context):
        for traj in request_iterator:
            self.buffer.add(*batch_proto2numpy(traj))
        print('buffer received Trajectories.')
        return apex_datatype_pb2.Nothing()

    def SendExperiences(self, request_iterator, context):
        for exp in request_iterator:
            self.buffer.add(*batch_proto2numpy(exp))
        print('buffer received Experiences.')
        return apex_datatype_pb2.Nothing()


def buffer(ip, port, buffer_args):
    for i in range(10):
        if check_port_in_use(port, ip):
            print(f'{i}: port {port} is under used.')
            time.sleep(1)
        else:
            break
    else:
        raise Exception('Cannot start learner correctly.')

    server = grpc.server(futures.ThreadPoolExecutor())
    apex_buffer_pb2_grpc.add_BufferServicer_to_server(BufferServicer(buffer_args), server)
    server.add_insecure_port(':'.join([ip, port]))
    server.start()
    print('start buffer success.')
    server.wait_for_termination()
