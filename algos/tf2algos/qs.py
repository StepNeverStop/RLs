import numpy as np
from utils.expl_expt import ExplorationExploitationClass
from utils.plot import ion, ioff, plot_heatmap


class QS:
    '''
    Q-learning/Sarsa/Expected Sarsa.
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,

                 mode='q',
                 lr=0.2,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_episode=100,
                 **kwargs):
        assert not hasattr(s_dim, '__len__')
        assert not is_continuous
        self.mode = mode
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.gamma = float(kwargs.get('gamma', 0.999))
        self.max_episode = int(kwargs.get('max_episode', 1000))
        self.step = 0
        self.episode = 0    # episode of now
        self.n_agents = int(kwargs.get('n_agents', 0))
        if self.n_agents <= 0:
            raise ValueError('agents num must larger than zero.')
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_episode=init2mid_annealing_episode,
                                                          max_episode=self.max_episode)
        self.table = np.zeros(shape=(self.s_dim, self.a_dim))
        self.lr = lr
        self.next_a = np.zeros(self.n_agents, dtype=np.int32)
        self.mask = []
        ion()

    def one_hot2int(self, x):
        idx = [np.where(np.asarray(i))[0][0] for i in x]
        return idx

    def partial_reset(self, done):
        self.mask = np.where(done)[0]

    def choose_action(self, s, visual_s=None, evaluation=False):
        s = self.one_hot2int(s)
        if self.mode == 'q':
            return self._get_action(s, evaluation)
        elif self.mode == 'sarsa' or self.mode == 'expected_sarsa':
            a = self._get_action(s, evaluation)
            self.next_a[self.mask] = a[self.mask]
            return self.next_a

    def _get_action(self, s, evaluation=False, _max=False):
        a = np.array([np.argmax(self.table[i, :]) for i in s])
        if not _max:
            if np.random.uniform() < self.expl_expt_mng.get_esp(self.episode, evaluation=evaluation):
                a = np.random.randint(0, self.a_dim, self.n_agents)
        return a

    def learn(self, **kwargs):
        self.episode = kwargs['episode']

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.step += 1
        s = self.one_hot2int(s)
        s_ = self.one_hot2int(s_)
        if self.mode == 'q':
            a_ = self._get_action(s_, _max=True)
            value = self.table[s_, a_]
        else:
            self.next_a = self._get_action(s_)
            if self.mode == 'expected_sarsa':
                value = np.mean(self.table[s_, :], axis=-1)
            else:
                value = self.table[s_, self.next_a]
        self.table[s, a] = (1 - self.lr) * self.table[s, a] + self.lr * (r + self.gamma * (1 - done) * value)
        if self.step % 1000 == 0:
            plot_heatmap(self.s_dim, self.a_dim, self.table)

    def close(self):
        ioff()

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        pass

    def __getattr__(self, x):
        # print(x)
        return lambda *args, **kwargs: 0
