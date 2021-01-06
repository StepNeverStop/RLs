from typing import Dict

from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color='red', bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return f'\x1b[{";".join(attr)}m{string}\x1b[0m'


def pwc(*args, color='red', bold=False, highlight=False):
    """
    Print with color
    """
    if isinstance(args, list) or isinstance(args, tuple):
        for s in args:
            print(colorize(s, color, bold, highlight))
    else:
        print(colorize(args, color, bold, highlight))


def show_dict(data: Dict):
    '''
    print the dictionary of configurations
    params:
        config: configurations of each variable
    '''
    logger.info('-' * 84)
    for k, v in data.items():
        logger.info(''.join([str(k).rjust(40), ' | ', str(v).ljust(40)]))
    logger.info('-' * 84)
