import grpc
import time
import threading

from concurrent import futures
from typing import Iterator

from rls.distribute.pb2 import (apex_datatype_pb2,
                                apex_buffer_pb2_grpc,
                                apex_learner_pb2_grpc)
from rls.distribute.utils.apex_utils import (proto2numpy,
                                             batch_proto2numpy,
                                             exps_and_prios2proto,
                                             proto2exps_and_tderror)
from rls.distribute.utils.check import check_port_in_use
from rls.memories.replay_buffer import PrioritizedExperienceReplay
from rls.utils.display import colorize
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class LearnThread(threading.Thread):

    def __init__(self, learner_ip, learner_port, buffer, lock):
        super().__init__()
        self.learner_channel = grpc.insecure_channel(':'.join([learner_ip, learner_port]))
        self.learner_stub = apex_learner_pb2_grpc.LearnerStub(self.learner_channel)
        self.buffer = buffer
        self.lock = lock

    def run(self):
        while True:
            if self.buffer.is_lg_batch_size:
                with self.lock:
                    exps, idxs = self.buffer.sample(return_index=True)
                    prios = self.buffer.get_IS_w().reshape(-1, 1)
                    td_error = self.learner_stub.SendExperienceGetPriorities(
                        exps_and_prios2proto(
                            exps=exps,
                            prios=prios))
                    td_error = proto2numpy(td_error)
                    self.buffer.update(td_error, idxs)
        self.learner_channel.close()


class BufferServicer(apex_buffer_pb2_grpc.BufferServicer):

    def __init__(self, buffer, lock):
        self.buffer = buffer
        self.lock = lock

    def SendTrajectories(self, request_iterator: Iterator[apex_datatype_pb2.ListNDarray], context) -> apex_datatype_pb2.Nothing:
        '''
        worker向buffer发送一批trajectory
        '''
        for traj in request_iterator:
            self.buffer.add(*batch_proto2numpy(traj))
        logger.info('receive Trajectories from worker.')
        return apex_datatype_pb2.Nothing()

    def SendExperiences(self, request_iterator: Iterator[apex_datatype_pb2.ExpsAndTDerror], context) -> apex_datatype_pb2.Nothing:
        '''
        worker向buffer发送一批经验
        '''
        with self.lock:
            for request in request_iterator:
                data, td_error = proto2exps_and_tderror(request)
                self.buffer.apex_add_batch(td_error, *data)
            logger.info('receive Experiences from worker.')
        return apex_datatype_pb2.Nothing()


def buffer(
        ip,
        port,
        learner_ip,
        learner_port,
        buffer_args):

    check_port_in_use(port, ip, try_times=10, server_name='buffer')

    buffer = PrioritizedExperienceReplay(**buffer_args)
    threadLock = threading.Lock()

    server = grpc.server(futures.ThreadPoolExecutor())
    apex_buffer_pb2_grpc.add_BufferServicer_to_server(BufferServicer(buffer=buffer, lock=threadLock), server)
    server.add_insecure_port(':'.join([ip, port]))
    server.start()
    logger.info(colorize('start buffer success.', color='green'))

    learn_thread = LearnThread(learner_ip, learner_port, buffer, threadLock)
    learn_thread.start()

    server.wait_for_termination()
