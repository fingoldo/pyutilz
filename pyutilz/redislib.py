# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed
ensure_installed("redis")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import redis
from time import sleep
from random import random
from redis.exceptions import ConnectionError

rc=None

def rconnect (redis_host:str, redis_port:int, redis_db_name:str, redis_db_pwd:str, decode_responses:bool=True):
    global rc
    rc = redis.Redis(host=redis_host, port=redis_port, db=redis_db_name, password=redis_db_pwd, decode_responses=decode_responses)
    
    return rc
    
def rexecute (method_name:str,*args,**kwargs):
    """
        Safely execute any Redis command, not worrying about temporarily network/server issues
        
    """
    res=None
    
    while True:
        try:
            method=getattr(rc,method_name)
        except Exception as e:
            logger.exception(e)
            sleep(1*random())
        else:
            break
            
    while True:
        try:
            res=method(*args,**kwargs)
        except ConnectionError as e:
            logger.exception(e)
            sleep(1*random())
        except Exception as e:
            logger.exception(e)            
        else:
            break    
    return res