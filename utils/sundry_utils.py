import sys
import logging


def create_logger(name, console_level, console_format, logger2file, file_name, file_level, file_format):
    logger = logging.Logger(name)
    logger.setLevel(level=console_level)
    stdout_handle = logging.StreamHandler(stream=sys.stdout)
    stdout_handle.setFormatter(logging.Formatter(console_format if console_level > 20 else '%(message)s'))
    logger.addHandler(stdout_handle)
    if logger2file:
        logfile_handle = logging.FileHandler(file_name)
        logfile_handle.setLevel(file_level)
        logfile_handle.setFormatter(logging.Formatter(file_format))
        logger.addHandler(logfile_handle)
    return logger


class LinearAnnealing:
    def __init__(self, x, x_, end):
        '''
        Params: 
            x: start value
            x_: end value
            end: annealing time
        '''
        assert end != 0, 'the time steps for annealing must larger than 0.'
        self.x = x
        self.x_ = x_
        self.interval = (x_ - x) / end

    def __call__(self, current):
        return max(self.x + self.interval * current, self.x_)
