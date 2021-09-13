

from torch.utils.tensorboard import SummaryWriter

from rls.utils.sundry_utils import check_or_create


class TensorboardLogger:

    def __init__(self,
                 log_dir,
                 ids=list(),
                 *args, **kwargs):

        self._ids = ids
        if len(ids) == 0:
            self._is_multi_logger = False
            check_or_create(log_dir, 'logs(summaries)')
            self._writer = SummaryWriter(log_dir)
        else:
            self._is_multi_logger = True
            self._writer = {}
            for id in ids:
                _log_dir = log_dir+f'_{id}'
                check_or_create(_log_dir, 'logs(summaries)')
                self._writer['id'] = SummaryWriter(_log_dir)

    def write(self, summaries, step, *args, **kwargs):
        if self._is_multi_logger:
            for id in self._ids:
                for k, v in summaries.get(id, {}).items():
                    self._writer[id].add_scalar(k, v, global_step=step)
        else:
            for k, v in summaries.items():
                self._writer.add_scalar(k, v, global_step=step)
