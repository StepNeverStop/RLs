import wandb


class WandbLogger:

    def __init__(self,
                 log_dir,
                 ids=list(),
                 training_name='test',
                 *args, **kwargs):
        wandb.login()
        wandb.init(project='RLs',
                   #    config=records_dict,
                   id=training_name,
                   resume='allow')
        self._is_multi_logger = True if len(ids) > 1 else False
        self._register_keys = {}

    def write(self, summaries, step, *args, **kwargs):
        if self._is_multi_logger:
            _summaries = {}
            for k, v in summaries.items():
                for _k, _v in v.items():
                    _summaries[k + '_' + _k] = _v
        else:
            _summaries = summaries
        self._check_is_registered(_summaries)
        _summaries.update({
            self._register_keys[list(_summaries.keys())[0]]: step
        })
        wandb.log(_summaries)

    def _check_is_registered(self, summaries):
        if list(summaries.keys())[0] in self._register_keys.keys():
            return
        else:
            step_metric = f'step{len(self._register_keys)}'
            self._register_keys.update({
                list(summaries.keys())[0]: step_metric
            })

            for k, v in summaries.items():
                wandb.run.define_metric(k, step_metric=step_metric)
