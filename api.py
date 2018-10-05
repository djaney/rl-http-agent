#!/usr/bin/env python3
import redis
from flask import Flask, request, jsonify
import os.path
import tensorflow as tf
import logging
import pickle
from core.environments import RedisEnv
from core.model_factory import get_agent, WEIGHTS


def load_weights(agent):
    weights_path = "{}/{}".format(dir_path, WEIGHTS)
    if os.path.isfile(weights_path) and os.access(weights_path, os.R_OK):
        print('save')
        agent.load_weights(weights_path)
    else:
        save_weights(agent)


def save_weights(agent):
    weights_path = "{}/{}".format(dir_path, WEIGHTS)
    agent.save_weights(weights_path, overwrite=True)
    print('save')
    os.chmod(weights_path, 0o777)


app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True
app.logger.disabled = True

CHANNEL = 'RedisEnv'
ACTION = 'RedisEnv_action'


dir_path = os.path.dirname(os.path.realpath(__file__))


env = RedisEnv(host='redis')
redis_object = redis.StrictRedis(host='redis', port=6379, password=None)

agent = get_agent()
graph = tf.get_default_graph()
load_weights(agent)


@app.route("/start", methods=['GET'])
def start():
    with graph.as_default():
        agent.fit(env, nb_steps=10000)

    return '', 204


@app.route("/get_status", methods=['GET'])
def get_status():
    return env.status


@app.route("/send_reset", methods=['POST'])
def send_reset():
    data = request.get_json(force=True)
    redis_object.publish(CHANNEL, pickle.dumps(data))
    return '', 204


@app.route("/get_action", methods=['GET'])
def get_action():
    action = pickle.loads(redis.get(ACTION))
    return action


@app.route("/step_result", methods=['POST'])
def step_result():
    data = request.get_json(force=True)
    tuple_data = (data[0], data[1], data[2], data[3])
    redis_object.publish(CHANNEL, pickle.dumps(tuple_data))
    return '', 204


app.run(debug=True, host='0.0.0.0')
