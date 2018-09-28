#!/usr/bin/env python3
import time
from flask import Flask, request, jsonify
from filelock import FileLock
import os.path
import tensorflow as tf
import random
import uuid
from core.model_factory import get_agent, WEIGHTS, LOCK

app = Flask(__name__)
pending = {}

dir_path = os.path.dirname(os.path.realpath(__file__))
weights_path = "{}/{}".format(dir_path, WEIGHTS)
lock_path = "{}/{}".format(dir_path, LOCK)

lock = FileLock(lock_path)
agent = get_agent()
graph = tf.get_default_graph()


@app.route("/forward", methods=['POST'])
def forward():
    data = request.get_json(force=True)
    assert ("state" in data.keys())

    state = data.get('state')

    with graph.as_default():
        action = agent.forward(state)

    while True:
        id = unique_id()
        if id not in pending.keys():
            break

    pending[id] = {"action": action, "state": state}

    return jsonify({"action": int(action), "id": str(id)})


@app.route("/backward", methods=['POST'])
def backward():
    data = request.get_json(force=True)
    assert ("id" in data.keys())
    assert ("reward" in data.keys())
    assert ("done" in data.keys())

    step = pending.get(data["id"])

    if not step:
        response = jsonify({"error": "id does not exist"})
        response.status_code = 400
        return response

    state = step["state"]
    action = step["action"]
    reward = data["reward"]
    done = data["done"]

    agent.remember(state, action, reward, done)
    return jsonify({})


def load_weights():
    with lock:
        if os.path.isfile(weights_path) and os.access(os.R_OK):
            agent.load_weights(weights_path)


def unique_id():
    return uuid.uuid1()
