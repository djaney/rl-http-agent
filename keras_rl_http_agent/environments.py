from gym import Env
import redis
import pickle

STATUS_PAUSE = 'STATUS_PAUSE'
STATUS_DONE = 'STATUS_DONE'
STATUS_WAITING_FOR_RESET_OBSERVATIONS = 'STATUS_WAITING_FOR_RESET_OBSERVATIONS'
STATUS_WAITING_FOR_ACTION_RESULT = 'STATUS_WAITING_FOR_ACTION_RESULT'


class RedisEnv(Env):

    def __init__(self, host, port=6379, password=None, channel_prefix='RedisEnv'):
        self._status = STATUS_DONE
        self.channel_reset_observation = channel_prefix + '_reset_observation'
        self.channel_step_result = channel_prefix + '_step_result'
        self.status_key = channel_prefix + '_status'
        self.action_key = channel_prefix + '_action'
        self.redis = redis.StrictRedis(host=host, port=port, password=password)
        self.redis.set(self.status_key, self._status)
        self.redis.set(self.action_key, pickle.dumps(0))
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe([self.channel_reset_observation, self.channel_step_result])

    def step(self, action):

        self.redis.set(self.action_key, pickle.dumps(action))
        ob, score, done, info = self._listen(STATUS_WAITING_FOR_ACTION_RESULT)

        return ob, score, done, info

    def reset(self):
        return self._listen(STATUS_WAITING_FOR_RESET_OBSERVATIONS)

    def render(self, mode='human'):
        pass

    def set_status_done(self):
        self._set_status(STATUS_DONE)

    @property
    def status(self):
        return self._status

    def _listen(self, status):
        self._set_status(status)

        if self._status == STATUS_WAITING_FOR_ACTION_RESULT:
            channel = self.channel_step_result
        elif self._status == STATUS_WAITING_FOR_RESET_OBSERVATIONS:
            channel = self.channel_reset_observation
        else:
            raise Exception("No channel for " + status)

        while True:
            for message in self.pubsub.listen():
                if message['type'] == 'message' and message['channel'].decode('utf-8') == channel:
                    self._set_status(STATUS_PAUSE)
                    return pickle.loads(message['data'])

    def _set_status(self, status):
        self._status = status
        self.redis.set(self.status_key, self._status)