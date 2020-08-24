from pathlib import Path
from os import path

# __file__ = "./__init__.py"
THIS_DIR = Path(path.dirname(path.abspath(__file__)))
PROJECT_DIR = THIS_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
