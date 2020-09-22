import grpc
import time
import numpy as np

from rls.distribute.pb2 import \
    apex_datatype_pb2, \
    apex_learner_pb2_grpc, \
    apex_buffer_pb2, \
    apex_buffer_pb2_grpc
from rls.distribute.utils.apex_utils import \
    numpy2proto, \
    batch_numpy2proto, \
    batch_proto2numpy
from rls.common.collector import GymCollector
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


def worker(learner_ip,
           learner_port,
           buffer_ip,
           buffer_port,
           model,
           env):
    learner_channel = grpc.insecure_channel(':'.join([learner_ip, learner_port]))
    buffer_channel = grpc.insecure_channel(':'.join([buffer_ip, buffer_port]))

    learner_stub = apex_learner_pb2_grpc.LearnerStub(learner_channel)
    buffer_stub = apex_buffer_pb2_grpc.BufferStub(buffer_channel)

    # arr = np.arange(8).reshape(2, 2, 2).astype(np.float32)
    # learner_stub.SendNumpyArray(numpy2proto(arr))

    # arr_list = [np.arange(4).reshape(2, 2), np.arange(3).astype(np.int32), np.array([])]
    # learner_stub.SendBatchNumpyArray(batch_numpy2proto(arr_list))

    while True:
        model.set_worker_params(
            batch_proto2numpy(learner_stub.GetParams(apex_datatype_pb2.Nothing())))
        for _ in range(10):
            buffer_stub.SendExperiences(GymCollector.run_exps_stream(env, model))
            time.sleep(0.5)
        # buffer_stub.SendTrajectories(GymCollector.run_trajectory(env, model))

    learner_channel.close()
    buffer_channel.close()
