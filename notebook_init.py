"""Repeatable code to init almost all my Jupyter notebooks ;-)"""

import os
import psutil
os.environ["POLARS_MAX_THREADS"] = str(max(1,int(psutil.cpu_count(logical=False)/2)))
print(f"Using {os.environ['POLARS_MAX_THREADS']} polars threads")
os.environ["_RJEM_MALLOC_CONF"] = "muzzy_decay_ms:500"

try:
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))
except: pass

%load_ext line_profiler
%load_ext autoreload
%load_ext autotime
%autoreload 2

from pyutilz.logginglib import init_logging

logger = init_logging( format="%(asctime)s - %(levelname)s - %(funcName)s-line:%(lineno)d - %(message)s")