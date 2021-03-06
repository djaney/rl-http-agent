import redis
from flask import Flask, request
import os.path
import tensorflow as tf
import logging
import pickle
from keras_rl_http_agent.environments import RedisEnv
from threading import Thread
from rl.callbacks import ModelIntervalCheckpoint


class FlaskApp:
    OBSERVATION = 'RedisEnv_observation'
    RESET_OBSERVATION = 'RedisEnv_reset_observation'
    STEP_RESULT = 'RedisEnv_step_result'
    ACTION = 'RedisEnv_action'
    STATUS = 'RedisEnv_status'

    def __init__(self, agent, weights_path, nb_steps=10000, checkpoint=100, redis_host='redis', redis_port=6379,
                 redis_password=None, verbose=2):
        self.weights_path = weights_path
        self.nb_steps = nb_steps
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.app = Flask(__name__)
        log = logging.getLogger('werkzeug')
        log.disabled = True
        self.app.logger.disabled = True
        self.app.add_url_rule('/start', 'start', self.start, methods=['GET'])
        self.app.add_url_rule('/get_status', 'get_status', self.get_status, methods=['GET'])
        self.app.add_url_rule('/send_reset', 'send_reset', self.send_reset, methods=['POST'])
        self.app.add_url_rule('/get_action', 'get_action', self.get_action, methods=['GET'])
        self.app.add_url_rule('/send_step_result', 'send_step_result', self.send_step_result, methods=['POST'])

        self.env = RedisEnv(host=redis_host, port=redis_port, password=redis_password)
        self.redis_object = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password)
        self.graph = tf.get_default_graph()

        self.agent = agent

        self.load_weights(self.agent, self.weights_path)

    def start(self):
        Thread(target=self.__start, args=(self.nb_steps, self.weights_path, self.agent,
                                          self.env, self.checkpoint, self.graph, self.verbose, self.__restart_env_status)).start()
        return '', 204

    def get_status(self):
        status = self.redis_object.get(self.STATUS)
        return status

    def send_reset(self):
        data = request.get_json(force=True)
        self.redis_object.publish(self.RESET_OBSERVATION, pickle.dumps(data))
        return '', 204

    def get_action(self):
        action = pickle.loads(self.redis_object.get(self.ACTION))
        return str(action)

    def send_step_result(self):
        data = request.get_json(force=True)
        tuple_data = (data[0], data[1], data[2], data[3])
        self.redis_object.publish(self.STEP_RESULT, pickle.dumps(tuple_data))
        return '', 204

    def __restart_env_status(self):
        self.env.set_status_done()

    @staticmethod
    def __start(nb_steps, weights_path, agent, environment, checkpoint, graph, verbose, callback=None):
        callbacks = ModelIntervalCheckpoint(filepath=weights_path, interval=checkpoint)
        with graph.as_default():
            agent.fit(environment, nb_steps=nb_steps, callbacks=[callbacks], verbose=verbose)
        if callback is not None:
            callback()

    @staticmethod
    def load_weights(agent, weights_path):
        if os.path.isfile(weights_path) and os.access(weights_path, os.R_OK):
            print('load')
            agent.load_weights(weights_path)

    def run(self, **kwargs):
        self.app.run(**kwargs)
