import grpc
import time
import numpy as np

from rls.distribute.pb2 import (apex_datatype_pb2,
                                apex_learner_pb2_grpc,
                                apex_buffer_pb2,
                                apex_buffer_pb2_grpc)
from rls.distribute.utils.apex_utils import (numpy2proto,
                                             batch_numpy2proto,
                                             batch_proto2numpy)
from rls.common.collector import GymCollector
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class WorkerCls(object):

    def __init__(self, env, model, worker_args, callback_func):
        self.env = env
        self.model = model
        self.callback_func = callback_func
        for k, v in worker_args.items():
            setattr(self, k, v)

    def run(self):
        while True:
            model.set_worker_params(self.callback_func())
            if self.is_send_traj:
                buffer_stub.SendTrajectories(GymCollector.run_trajectory(env, model))
            else:
                for _ in range(10):
                    buffer_stub.SendExperiences(GymCollector.run_exps_stream(env, model))
            time.sleep(self.rollout_interval)


def worker(env,
           model,
           learner_ip,
           learner_port,
           buffer_ip,
           buffer_port,
           worker_args):
    learner_channel = grpc.insecure_channel(':'.join([learner_ip, learner_port]))
    buffer_channel = grpc.insecure_channel(':'.join([buffer_ip, buffer_port]))

    learner_stub = apex_learner_pb2_grpc.LearnerStub(learner_channel)
    buffer_stub = apex_buffer_pb2_grpc.BufferStub(buffer_channel)

    # arr = np.arange(8).reshape(2, 2, 2).astype(np.float32)
    # learner_stub.SendNumpyArray(numpy2proto(arr))

    # arr_list = [np.arange(4).reshape(2, 2), np.arange(3).astype(np.int32), np.array([])]
    # learner_stub.SendBatchNumpyArray(batch_numpy2proto(arr_list))

    workercls = WorkerCls(env, model, worker_args, callback_func=lambda: batch_proto2numpy(learner_stub.GetParams(apex_datatype_pb2.Nothing())))
    workercls.run()

    learner_channel.close()
    buffer_channel.close()
