#!/usr/bin/env python3
import time
from flask import Flask, request, jsonify
from filelock import FileLock
import os.path
import tensorflow as tf
import uuid

from core.model_factory import get_agent, WEIGHTS, LOCK


def load_weights():
    weights_path = "{}/{}".format(dir_path, WEIGHTS)
    lock_path = "{}/{}".format(dir_path, LOCK)
    lock = FileLock(lock_path)
    with lock:
        if os.path.isfile(weights_path) and os.access(os.R_OK):
            agent.load_weights(weights_path)


def save_weights():
    weights_path = "{}/{}".format(dir_path, WEIGHTS)
    lock_path = "{}/{}".format(dir_path, LOCK)
    lock = FileLock(lock_path)
    with lock:
        if os.path.isfile(weights_path) and os.access(os.R_OK):
            agent.save_weights(weights_path)


def unique_id():
    return uuid.uuid1()


app = Flask(__name__)
# log = logging.getLogger('werkzeug')
# log.disabled = True
# app.logger.disabled = True


pending = {}
sessions = {}

dir_path = os.path.dirname(os.path.realpath(__file__))


agent = get_agent()
graph = tf.get_default_graph()
# load_weights()


@app.route("/forward", methods=['POST'])
def forward():
    data = request.get_json(force=True)
    assert ("state" in data.keys())

    state = data.get('state')

    with graph.as_default():
        action = agent.forward(state)

    while True:
        id = str(unique_id())
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
    assert ("session_id" in data.keys())

    session_id = data['session_id']

    step = pending.get(data["id"])

    if not step:
        response = jsonify({"error": "id does not exist"})
        response.status_code = 400
        return response

    state = step["state"]
    action = step["action"]
    reward = data["reward"]
    done = data["done"]

    if session_id not in sessions:
        sessions[session_id] = []
    session = sessions[session_id]
    session.append((state, action, reward, done))

    if done:
        if session:

            for state, action, reward, done in session:
                agent.forward(state)
                agent.backward(reward)

            del sessions[session_id]
            save_weights()

    return jsonify({})



