#!/usr/bin/env python3
import redis
from flask import Flask, request, jsonify
import os.path
import tensorflow as tf
import logging
import pickle
from core.environments import RedisEnv
from core.model_factory import get_agent, WEIGHTS
from threading import Thread
from rl.callbacks import ModelIntervalCheckpoint


def load_weights(agent):
    weights_path = "{}/{}".format(dir_path, WEIGHTS)
    if os.path.isfile(weights_path) and os.access(weights_path, os.R_OK):
        print('load')
        agent.load_weights(weights_path)


def __start(a, e, g):
    weights_path = "{}/{}".format(dir_path, WEIGHTS)
    callbacks = ModelIntervalCheckpoint(filepath=weights_path, interval=1000)
    with g.as_default():
        a.fit(e, nb_steps=10000, callbacks=[callbacks], verbose=2)


app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True
app.logger.disabled = True

OBSERVATION = 'RedisEnv_observation'
RESET_OBSERVATION = 'RedisEnv_reset_observation'
STEP_RESULT = 'RedisEnv_step_result'
ACTION = 'RedisEnv_action'
STATUS = 'RedisEnv_status'


dir_path = os.path.dirname(os.path.realpath(__file__))


env = RedisEnv(host='redis')
redis_object = redis.StrictRedis(host='redis', port=6379, password=None)
graph = tf.get_default_graph()

with graph.as_default():
    agent = get_agent()

load_weights(agent)


@app.route("/start", methods=['GET'])
def start():
    Thread(target=__start, args=(agent, env, graph)).start()
    return '', 204


@app.route("/get_status", methods=['GET'])
def get_status():
    status = redis_object.get(STATUS)
    return status


@app.route("/send_reset", methods=['POST'])
def send_reset():
    data = request.get_json(force=True)
    redis_object.publish(RESET_OBSERVATION, pickle.dumps(data))
    return '', 204


@app.route("/get_action", methods=['GET'])
def get_action():
    action = pickle.loads(redis_object.get(ACTION))
    return str(action)


@app.route("/send_step_result", methods=['POST'])
def send_step_result():
    data = request.get_json(force=True)
    tuple_data = (data[0], data[1], data[2], data[3])
    redis_object.publish(STEP_RESULT, pickle.dumps(tuple_data))
    return '', 204


app.run(debug=True, host='0.0.0.0')
