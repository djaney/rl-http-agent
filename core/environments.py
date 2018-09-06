from gym import Env
import numpy as np


class SocketEnv(Env):
    """
    state payload: command (np.int8 1b) + score (np.float16 2b) + done (np.int8 1b) + state (state size * np.float16 2b)
    action payload: command (np.int8 1b) + action (action size * np.float16 2b)
    """
    MSG_PAIR = 1
    MSG_STATE = 2
    MSG_ACTION = 3

    def __init__(self, socket, state_size, action_size, dtype=np.float16):
        self.socket = socket
        self.state_size = state_size
        self.action_size = action_size
        self.dtype = dtype
        # payload_size = command + score + done + state
        self.state_payload_size = 4 + self.state_size * np.dtype(self.dtype).itemsize
        self.action_payload_size = 1 + self.action_size * np.dtype(self.dtype).itemsize
        self.pair = None
        # pair
        while True:
            data, address = self.socket.recvfrom(1)
            # do not accept wrong message type
            if data[0] == self.MSG_PAIR:
                self.pair = address
                payload = np.asarray([self.MSG_PAIR], dtype=np.int8).tobytes() + \
                          np.asarray([self.state_payload_size], dtype=np.int8).tobytes() + \
                          np.asarray([self.action_payload_size], dtype=np.int8).tobytes()
                self.socket.sendto(payload, self.pair)
                break

    def step(self, action):
        """
        :type action: numpy.ndarray
        """
        assert len(action) == self.action_size
        assert self.pair is not None

        payload = np.asarray([self.MSG_ACTION], dtype=np.int8).tobytes() + \
                  np.asarray(action, dtype=self.dtype).tobytes()
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
