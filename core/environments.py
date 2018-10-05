from gym import Env
import redis
import pickle

STATUS_RESTING = 'STATUS_RESTING'
STATUS_WAITING_FOR_OBSERVATIONS = 'STATUS_WAITING_FOR_OBSERVATIONS'
STATUS_WAITING_FOR_RESET_OBSERVATIONS = 'STATUS_WAITING_FOR_RESET_OBSERVATIONS'
STATUS_WAITING_FOR_ACTION_RESULT = 'STATUS_WAITING_FOR_ACTION_RESULT'


class RedisEnv(Env):

    def __init__(self, host, port=6379, password=None, channel='RedisEnv'):
        self._status = STATUS_RESTING
        self.channel = channel + '_observation'
        self.status_key = channel + '_status'
        self.action_key = channel + '_action'
        self.redis = redis.StrictRedis(host=host, port=port, password=password)
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe(self.channel)

    def step(self, action):

        self.redis.set(self.action_key, pickle.dumps(action))
        ob, score, done, info = self._listen(STATUS_WAITING_FOR_ACTION_RESULT)

        return ob, score, done, info

    def reset(self):
        return self._listen(STATUS_WAITING_FOR_RESET_OBSERVATIONS)

    def render(self, mode='human'):
        pass

    @property
    def status(self):
        return self._status

    def _listen(self, status):
        self._status = status
        for message in self.pubsub.listen():
            if message['data'] == 1:
                continue
            return pickle.loads(message['data'])

