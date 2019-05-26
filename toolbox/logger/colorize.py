
__author__ = 'Hyunsoo Park (hspark8312@ncsoft.com), Game AI Lab, NCSOFT'

import logging
import platform
from functools import wraps


class Color(object):
    """
    텍스트 컬러 출력용

    Windows에서는 VT100 활성화 필요
    """

    enable = True

    RESET = "\033[0m"
    BLACK = "\033[30m"
    WHITE = "\033[37m"
    CYAN = "\033[36m"
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def header(obj):
        return '===== ' + str(obj) + ' ====='

    @staticmethod
    def debug(obj):
        return f'[{Color.BLUE}DEBUG{Color.ENDC}] {str(obj)}'

    @staticmethod
    def info(obj):
        return f'[{Color.GREEN}INFO{Color.ENDC}] {str(obj)}'

    @staticmethod
    def notice(obj):
        return f'[{Color.PURPLE}NOTICE{Color.ENDC}] {str(obj)}'

    @staticmethod
    def warning(obj):
        return f'[{Color.YELLOW}WARNING{Color.ENDC}] {str(obj)}'

    @staticmethod
    def error(obj):
        return f'[{Color.RED}ERROR{Color.ENDC}] {str(obj)}'

    @staticmethod
    def purple(obj):
        if Color.enable:
            return Color.PURPLE + str(obj) + Color.ENDC
        else:
            return str(obj)

    @staticmethod
    def blue(obj):
        if Color.enable:
            return Color.BLUE + str(obj) + Color.ENDC
        else:
            return str(obj)

    @staticmethod
    def green(obj):
        if Color.enable:
            return Color.GREEN + str(obj) + Color.ENDC
        else:
            return str(obj)

    @staticmethod
    def yellow(obj):
        if Color.enable:
            return Color.YELLOW + str(obj) + Color.ENDC
        else:
            return str(obj)

    @staticmethod
    def red(obj):
        if Color.enable:
            return Color.RED + str(obj) + Color.ENDC
        else:
            return str(obj)

    @staticmethod
    def bold(obj):
        if Color.enable:
            return Color.BOLD + str(obj) + Color.ENDC
        else:
            return str(obj)


class ColoredFormatter(logging.Formatter):

    COLORS = dict(
        WARNING=Color.yellow,
        INFO=Color.green,
        DEBUG=Color.blue,
        CRITICAL=Color.yellow,
        ERROR=Color.red
    )

    def __init__(self, fmt):
        logging.Formatter.__init__(self, fmt=fmt)

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = self.COLORS[levelname](f'{levelname:7}')
        return logging.Formatter.format(self, record)


if platform.system() == 'Windows':
    Color.enable = False


if __name__ == '__main__':

    print(Color.enable)
    Color.enable = True

    print(Color.purple('HEADER'))
    print(Color.blue('BLUE'))
    print(Color.green('GREEN'))
    print(Color.yellow('WARNING'))
    print(Color.red('FAIL'))

    def underline(func):
        @wraps(func)
        def wrapped(text):
            if Color.enable:
                text = Color.UNDERLINE + text + Color.ENDC
            func(text)

        return wrapped

    @underline
    def print2(text):
        print(text)

    print2('RED')
