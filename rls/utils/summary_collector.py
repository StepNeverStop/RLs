from collections import defaultdict

import numpy as np

from rls.utils.converter import to_numpy_or_number


class SummaryCollector:
    MIN = 0
    MAX = 1
    MEAN = 2
    ALL = 3

    def __init__(self, mode=ALL):
        self._mode = mode
        self.summaries_dict = defaultdict(float)

    def add(self, scope, key, value):
        name = '/'.join([scope, key])
        self.summaries_dict[name] = value

    def fetch(self):
        ret_dict = to_numpy_or_number(self.summaries_dict)

        for k in self.summaries_dict.keys():
            v = ret_dict[k]
            if isinstance(v, (np.ndarray, tuple, list)):
                if self._mode == SummaryCollector.MIN:
                    ret_dict[k + '_min'] = np.min(v)
                elif self._mode == SummaryCollector.MAX:
                    ret_dict[k + '_max'] = np.max(v)
                elif self._mode == SummaryCollector.MEAN:
                    ret_dict[k + '_mean'] = np.mean(v)
                elif self._mode == SummaryCollector.ALL:
                    ret_dict[k + '_min'] = np.min(v)
                    ret_dict[k + '_max'] = np.max(v)
                    ret_dict[k + '_mean'] = np.mean(v)
                else:
                    raise NotImplementedError
                del ret_dict[k]

        self.reset()
        return ret_dict

    def reset(self):
        self.summaries_dict = defaultdict(float)
