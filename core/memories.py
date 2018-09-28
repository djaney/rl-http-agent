from rl.memory import Memory, Experience
import redis
import numpy as np
import random
import pickle


class RedisMemory(Memory):

    conn = None
    name = "memory"

    def connect(self, host="redis", port=6379, password=None):

        self.conn = redis.StrictRedis(
            host=host,
            port=port,
            password=password)

        if not self.conn:
            raise Exception("Cannot connect to {}:{}".format(host, port))

    def sample(self, batch_size, batch_idxs=None):
        mem_size = self.conn.llen(self.name)

        if mem_size < 2:
            return []

        sample = []
        while len(sample) < batch_size:
            start = random.randint(0, mem_size - 2)
            pair = self.conn.lrange(self.name, start, start + 1)
            if len(pair) < 2:
                return []
            else:
                item = self.from_bytes(pair[0])
                next_item = self.from_bytes(pair[1])

                sample.append(Experience(state0=item[0],
                                         action=item[1], reward=item[2], state1=next_item[0], terminal1=item[3]))

        return sample

    def append(self, observation, action, reward, terminal, training=True):
        entry_bytes = self.to_bytes([observation, action, reward, terminal])
        self.conn.lpush(self.name, entry_bytes)

    def get_recent_state(self, current_observation):
        start = -self.window_length
        end = -1
        # get range
        recent_list = self.conn.lrange(self.name, start, end)
        # convert to types
        recent_list = [self.from_bytes(i) for i in recent_list]
        list_len = len(recent_list)
        if list_len == 0:
            # if no memory repeat input
            recent_list = [current_observation] * self.window_length
            list_len = self.window_length
        else:
            # extract state from memory
            recent_list = [i[0] for i in recent_list]
        if list_len < self.window_length:
            # if memory not enough repeat first element
            recent_list = [recent_list[0]] * (self.window_length-list_len) + recent_list

        return recent_list

    @staticmethod
    def from_bytes(inp):
        return pickle.loads(inp)

    @staticmethod
    def to_bytes(inp):
        return pickle.dumps(inp)
