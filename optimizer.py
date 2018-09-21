#!/usr/bin/env python3
import time
import filelock
from core.model_factory import get_agent, WEIGHTS, LOCK
import os.path
from filelock import Timeout, FileLock


def main():

    lock = FileLock(LOCK)

    model = get_agent()
    with lock:
        if os.path.isfile(WEIGHTS) and os.access(WEIGHTS, os.W_OK & os.R_OK):
            print("weights loaded")
            model.load_weights(WEIGHTS)
        else:
            print("weights saved")
            model.save_weights(WEIGHTS)
        os.chmod(WEIGHTS, 0o777)
        os.chmod(LOCK, 0o777)

    while True:
        print('tick')
        time.sleep(1)

print(__name__)
if __name__ == "__main__":
    main()