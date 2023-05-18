import os
import sys

py_dir = os.path.split(os.getcwd())[0]

if py_dir not in sys.path:
    sys.path.append(py_dir)
