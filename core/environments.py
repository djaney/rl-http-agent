from gym import Env
import numpy as np


class SocketEnv(Env):
    MSG_PAIR = 1
    MSG_STATE = 2
    MSG_ACTION = 3

    def __init__(self, socket, state_size, action_size, dtype=np.float16):
        self.socket = socket
        self.state_size = state_size
        self.action_size = action_size
        self.dtype = dtype
        # payload_size = command + score + done + state
        self.state_payload_size = 1 + 2 + 1 + self.state_size * 2
        self.pair = None
        # pair
        while True:
            data, address = self.socket.recvfrom(self.state_payload_size)
            # do not accept wrong message type
            if data[0] == self.MSG_PAIR:
                self.pair = address
                break

    def step(self, action):
        """
        :type action: numpy.ndarray
        """
        assert len(action) == self.action_size
        assert self.pair is not None

        payload = np.asarray([self.MSG_ACTION], dtype=np.int8).tobytes() + action.tobytes()
        self.socket.sendto(payload, self.pair)
        ob, score, done = self._get_state()
        return ob, score, done, {}

    def reset(self):
        ob, score, done = self._get_state()
        return ob

    def render(self, mode='human'):
        pass

    def _get_state(self):

        while True:
            data = self.socket.recv(self.state_payload_size)
            # do not accept wrong message type
            if data[0] == self.MSG_STATE:
                return (
                    np.frombuffer(data[4:], dtype=self.dtype),  # state
                    np.frombuffer(data[1:3], dtype=np.float16),  # score
                    np.frombuffer(data[3:4], dtype=np.int8),  # done
                )
