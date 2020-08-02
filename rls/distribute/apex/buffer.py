import grpc
import time

from concurrent import futures
from typing import Iterator

from rls.distribute.pb2 import \
    apex_datatype_pb2, \
    apex_buffer_pb2_grpc, \
    apex_learner_pb2_grpc
from rls.distribute.utils.apex_utils import \
    proto2numpy, \
    batch_proto2numpy, \
    exps_and_prios2proto
from rls.distribute.utils.check import check_port_in_use
from rls.memories.replay_buffer import PrioritizedExperienceReplay
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class BufferServicer(apex_buffer_pb2_grpc.BufferServicer):

    def __init__(self, buffer):
        self.buffer = buffer

    def SendTrajectories(self, request_iterator: Iterator[apex_datatype_pb2.ListNDarray], context) -> apex_datatype_pb2.Nothing:
        for traj in request_iterator:
            self.buffer.add(*batch_proto2numpy(traj))
        logger.info('receive Trajectories from worker.')
        return apex_datatype_pb2.Nothing()

    def SendExperiences(self, request_iterator: Iterator[apex_datatype_pb2.ListNDarray], context) -> apex_datatype_pb2.Nothing:
        for exp in request_iterator:
            self.buffer.add(*batch_proto2numpy(exp))
        logger.info('receive Experiences from worker.')
        return apex_datatype_pb2.Nothing()


def buffer(
        ip,
        port,
        learner_ip,
        learner_port,
        buffer_args):
    for i in range(10):
        if check_port_in_use(port, ip):
            print(f'{i}: port {port} is under used.')
            time.sleep(1)
        else:
            break
    else:
        raise Exception('Cannot start learner correctly.')

    buffer = PrioritizedExperienceReplay(**buffer_args)

    server = grpc.server(futures.ThreadPoolExecutor())
    apex_buffer_pb2_grpc.add_BufferServicer_to_server(BufferServicer(buffer=buffer), server)
    server.add_insecure_port(':'.join([ip, port]))
    server.start()
    print('start buffer success.')

    learner_channel = grpc.insecure_channel(':'.join([learner_ip, learner_port]))
    learner_stub = apex_learner_pb2_grpc.LearnerStub(learner_channel)
    while True:
        if buffer.is_lg_batch_size:
            exps = buffer.sample()
            prios = buffer.get_IS_w().reshape(-1, 1)
            td_error = learner_stub.SendExperienceGetPriorities(
                exps_and_prios2proto(
                    exps=exps,
                    prios=prios))
            td_error = proto2numpy(td_error)
            buffer.update(td_error, 0)

    server.wait_for_termination()
    learner_channel.close()
