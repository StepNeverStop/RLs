import logging

CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

_loggers = set()
_log_level = NOTSET
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Create a logger with the specified name. The logger will use the log level
    specified by set_log_level()
    """
    logger = logging.getLogger(name=name)

    # If we've already set the log level, make sure new loggers use it
    if _log_level != NOTSET:
        logger.setLevel(_log_level)

    # Keep track of this logger so that we can change the log level later
    _loggers.add(logger)
    return logger


def set_log_level(log_level: int) -> None:
    """
    Set the logging level. This will also configure the logging format (if it hasn't already been set).
    """
    global _log_level
    _log_level = log_level

    # Configure the log format.
    # In theory, this would be sufficient, but if another library calls logging.basicConfig
    # first, it doesn't have any effect.
    if _log_level > INFO:
        logging.basicConfig(level=_log_level, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    else:
        logging.basicConfig(level=_log_level, format='%(message)s')

    for logger in _loggers:
        logger.setLevel(log_level)


def set_log_file(log_file: str = None) -> None:
    if log_file:
        for logger in _loggers:
            logfile_handle = logging.FileHandler(log_file)
            logfile_handle.setLevel(_log_level)
            logfile_handle.setFormatter(logging.Formatter(LOG_FORMAT))
            logger.addHandler(logfile_handle)
    else:
        pass