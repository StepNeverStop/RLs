from .none import NoneLogger
from .tensorboard import TensorboardLogger
from .wandb import WandbLogger

Log_REGISTER = {}

Log_REGISTER['none'] = NoneLogger
Log_REGISTER['tensorboard'] = TensorboardLogger
Log_REGISTER['wandb'] = WandbLogger
