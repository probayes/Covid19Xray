from pathlib import Path
from os import path
import logging
from joblib import Memory

logger_level = logging.INFO

# __file__ = "./__init__.py"
THIS_DIR = Path(path.dirname(path.abspath(__file__)))
PROJECT_DIR = THIS_DIR.parent

# logger
logging.getLogger('pyprobayes').addHandler(logging.NullHandler())

