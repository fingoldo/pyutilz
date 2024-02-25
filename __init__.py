from .version import __version__

from enum import Enum

class NotebookColors(str,Enum):
    red = '\033[91m'
    green = '\033[92m'
    yellow = '\033[93m'
    blue = '\033[94m'
    pink = '\033[95m'
    teal = '\033[96m'
    grey = '\033[97m'
