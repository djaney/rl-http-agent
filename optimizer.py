#!/usr/bin/env python3
import time
import filelock
from core.model_factory import get_agent, WEIGHTS, LOCK
import os.path
from filelock import FileLock

dir_path = os.path.dirname(os.path.realpath(__file__))
weights_path = "{}/{}".format(dir_path, WEIGHTS)
lock_path = "{}/{}".format(dir_path, LOCK)

def main():

    lock = FileLock(lock_path)
    agent = get_agent()
    
    while True:

        with lock:
            os.chmod(lock_path, 0o777)
            if os.path.isfile(weights_path) and os.access(weights_path, os.W_OK & os.R_OK):
                agent.load_weights(weights_path)
                agent.optimize()
            else:
                agent.save_weights(weights_path)
            
            time.sleep(1)

            agent.save_weights(weights_path, overwrite=True)
            os.chmod(weights_path, 0o777)
        time.sleep(1)
        
if __name__ == "__main__":
    main()