from .none import NoneLogger
from .tensorboard import TensorboardLogger

Log_REGISTER = {}

Log_REGISTER['none'] = NoneLogger
Log_REGISTER['tensorboard'] = TensorboardLogger
try:
    from .wandb import WandbLogger

    Log_REGISTER['wandb'] = WandbLogger
except:
    pass
