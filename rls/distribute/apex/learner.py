import grpc
import time
import threading
import numpy as np

from rls.utils.np_utils import \
    SMA, \
    arrprint

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
from rls.common.collector import GymCollector
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class EvalThread(threading.Thread):

    def __init__(self, env, model, lock):
        super().__init__()
        self.env = env
        self.model = model
        self.lock = lock

    def run(self):
        n = self.env.n
        i = 1 if self.env.obs_type == 'visual' else 0
        state = [np.array([[]] * n), np.array([[]] * n)]
        sma = SMA(100)
        total_step = 0
        episode = 0

        while True:
            self.lock.acquire()
            episode += 1
            self.model.reset()
            state[i] = self.env.reset()
            dones_flag = np.zeros(self.env.n)
            step = 0
            rets = np.zeros(self.env.n)
            last_done_step = -1
            while True:
                step += 1
                # env.render(record=False)
                action = self.model.choose_action(s=state[0], visual_s=state[1])
                _, reward, done, info, state[i] = self.env.step(action)
                rets += (1 - dones_flag) * reward
                dones_flag = np.sign(dones_flag + done)
                self.model.partial_reset(done)
                total_step += 1
                if all(dones_flag):
                    if last_done_step == -1:
                        last_done_step = step
                    break

                if step >= 200:
                    break

            sma.update(rets)
            self.model.writer_summary(
                episode,
                reward_mean=rets.mean(),
                reward_min=rets.min(),
                reward_max=rets.max(),
                step=last_done_step,
                **sma.rs
            )
            print(f'Eps: {episode:3d} | S: {step:4d} | LDS {last_done_step:4d} | R: {arrprint(rets, 2)}')
            self.lock.release()
            time.sleep(5)


class LearnerServicer(apex_learner_pb2_grpc.LearnerServicer):

    def __init__(self, model, lock):
        self.model = model
        self.train_step = 0
        self.lock = lock

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
        # logger.info('send params to worker.')
        return params

    def SendExperienceGetPriorities(self, request: apex_datatype_pb2.ExpsAndPrios, context) -> apex_datatype_pb2.NDarray:
        self.lock.acquire()
        data, prios = proto2exps_and_prios(request)
        td_error = numpy2proto(self.model.apex_learn(self.train_step, data, prios))
        self.train_step += 1
        if self.train_step % 100 == 0:
            self.model.save_checkpoint(train_step=self.train_step)
        # logger.info('send new priorities to buffer.')
        self.lock.release()
        return td_error


def learner(ip, port, model, env):
    check_port_in_use(port, ip, try_times=10, server_name='learner')
    assert hasattr(model, 'apex_learn'), 'this algorithm does not support Ape-X learning for now.'

    thread_lock = threading.Lock()

    server = grpc.server(futures.ThreadPoolExecutor())
    apex_learner_pb2_grpc.add_LearnerServicer_to_server(LearnerServicer(model, thread_lock), server)
    server.add_insecure_port(':'.join([ip, port]))
    server.start()
    logger.info('start learner success.')

    eval_thread = EvalThread(env, model, thread_lock)
    eval_thread.start()

    # GymCollector.evaluate(env, model)
    server.wait_for_termination()
