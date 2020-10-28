import grpc
import time
import numpy as np

from rls.utils.np_utils import (SMA,
                                arrprint)
from rls.distribute.pb2 import (apex_datatype_pb2,
                                apex_learner_pb2_grpc)
from rls.distribute.utils.apex_utils import batch_proto2numpy
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class EvalProc(object):
    '''
    评估策略性能
    '''

    def __init__(self, env, model, evaluator_args, callback_func):
        super().__init__()
        self.env = env
        self.model = model
        self.callback_func = callback_func

        for k, v in evaluator_args.items():
            setattr(self, k, v)

    def run(self):
        n = self.env.n
        i = 1 if self.env.obs_type == 'visual' else 0
        state = [np.full((n, 0), []), np.full((n, 0), [])]
        sma = SMA(100)
        total_step = 0
        episode = 0

        while True:
            if episode % self.pull_interval:
                self.model.set_worker_params(self.callback_func())
                logger.info('pull parameters from success.')
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
            logger.info(f'Eps: {episode:3d} | S: {step:4d} | LDS {last_done_step:4d} | R: {arrprint(rets, 2)}')
            time.sleep(self.episode_sleep)


def evaluator(env,
              model,
              learner_ip,
              learner_port,
              evaluator_args):
    learner_channel = grpc.insecure_channel(':'.join([learner_ip, learner_port]))
    learner_stub = apex_learner_pb2_grpc.LearnerStub(learner_channel)

    evalproc = EvalProc(env, model, evaluator_args, callback_func=lambda: batch_proto2numpy(learner_stub.GetParams(apex_datatype_pb2.Nothing())))
    evalproc.run()

    learner_channel.close()
