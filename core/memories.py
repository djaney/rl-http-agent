from rl.memory import Memory, Experience
import redis
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
            start = random.randint(0, mem_size - self.window_length)
            window = self.conn.lrange(self.name, start, start + self.window_length+1)
            if len(window) < self.window_length:
                return []
            else:
                state0 = []
                state1 = []
                reward = []
                action = []
                terminal = []
                for i in range(self.window_length):
                    item = self.from_bytes(window[i])
                    next_item = self.from_bytes(window[i+1])
                    state0.append(item[0])
                    state1.append(next_item[0])
                    reward.append(item[2])
                    action.append(item[1])
                    terminal.append(item[3])

                sample.append(Experience(state0=state0, action=action[-1], reward=reward[-1], state1=state1, terminal1=terminal[-1]))

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
