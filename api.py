#!/usr/bin/env python3
import time
from flask import Flask
from filelock import FileLock
import os.path


app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
weights_path = "{}/{}".format(dir_path, WEIGHTS)
lock_path = "{}/{}".format(dir_path, LOCK)

lock = FileLock(lock_path)
agent = get_agent()

@app.route("/forward")
def step(methods=['POST']):
    """get next step"""
    return "Hello World!"

@app.route("/backward")
def reward(methods=['POST']):
    """set reward"""
    return "Hello World!"

def load_weights():
    with lock:
        if os.path.isfile(weights_path) and os.access(os.R_OK):
            agent.load_weights(weights_path)