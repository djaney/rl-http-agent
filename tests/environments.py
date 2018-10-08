import unittest

from keras_rl_http_agent.environments import SocketEnv
import socket
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import Policy
from keras.optimizers import Adam


def start_environment(sock, response):
    env = SocketEnv(sock, 7)
    ob = env.reset()
    response += ob


class TestPolicy(Policy):

    def select_action(self, q_values):
        action = np.argmax(q_values)
        return np.asarray([action], np.float16)


class TestSocketEnv(unittest.TestCase):

    def setUp(self):
        self.response = []
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_sock.bind(('', 0))
        self.server_sock_address = self.server_sock.getsockname()

        self.pair_payload = np.asarray([SocketEnv.MSG_PAIR], dtype=np.int8).tobytes()

    def _prepare_state_payload(self, cmd, state, score, done):
        return np.asarray([cmd], dtype=np.int8).tobytes() + score.tobytes() + done.tobytes() + state.tobytes()

    def tearDown(self):
        self.server_sock.close()

    def test_cycle(self):
        # prepare payload
        score_message = np.asarray([100], dtype=np.float16)
        state_message = np.asarray([1.1, 1.2, 1.3], dtype=np.float16)
        done_message = np.asarray([0], dtype=np.int8)
        state_payload = self._prepare_state_payload(SocketEnv.MSG_STATE, state_message, score_message, done_message)

        action_message = np.asarray([1, 0], dtype=np.float16)

        # create client socket
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # send pair request
        client_sock.sendto(self.pair_payload, self.server_sock_address)

        # recv pair request
        env = SocketEnv(self.server_sock, 3, 2, dtype=np.float16)
        client_sock.recv(3) # ack pair

        # send state payload
        client_sock.sendto(state_payload, self.server_sock_address)
        ob = env.reset()
        np.testing.assert_array_equal(state_message, ob)

        for _ in range(1):
            # send state
            client_sock.sendto(state_payload, self.server_sock_address)
            ob, score, done, info = env.step(action_message)
            res = client_sock.recv(5)
            cmd = res[0]
            action_res = np.frombuffer(res[1:], dtype=np.float16)
            np.testing.assert_array_equal([SocketEnv.MSG_ACTION], cmd)
            np.testing.assert_array_equal(action_message, action_res)
            np.testing.assert_array_equal(state_message, ob)
            np.testing.assert_array_equal(score_message, score)
            np.testing.assert_array_equal(done_message, done)

        # close client
        client_sock.close()

    def test_keras_rl(self):
        """
        Keras-rl lifecycle
        client --- env
        pair -->
        RESET
        --> state
        ITERATE
        --> state
        <-- action
        ...
        --> state with done
        TRAIN
        back to RESET
        """
        score_message = np.asarray([100], dtype=np.float16)
        state_message = np.asarray([1.1, 1.2, 1.3], dtype=np.float16)
        done_message = np.asarray([0], dtype=np.int8)
        state_payload = self._prepare_state_payload(SocketEnv.MSG_STATE, state_message, score_message, done_message)
        state_payload2 = self._prepare_state_payload(SocketEnv.MSG_STATE, state_message, score_message,
                                                     np.asarray([1], dtype=np.int8))

        client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_sock.sendto(self.pair_payload, self.server_sock_address)

        env = SocketEnv(self.server_sock, 3, 1, dtype=np.float16)
        client_sock.recv(3) # ack pair

        model = Sequential()
        model.add(Flatten(input_shape=(1, 3,)))
        model.add(Dense(1))

        memory = SequentialMemory(limit=50000, window_length=1)
        policy = TestPolicy()
        dqn = DQNAgent(model=model, nb_actions=1, memory=memory, policy=policy)
        dqn.compile(Adam())

        client_sock.sendto(state_payload, self.server_sock_address)
        client_sock.sendto(state_payload, self.server_sock_address)
        client_sock.sendto(state_payload, self.server_sock_address)
        client_sock.sendto(state_payload, self.server_sock_address)
        client_sock.sendto(state_payload, self.server_sock_address)
        client_sock.sendto(state_payload2, self.server_sock_address)

        dqn.fit(env, nb_steps=5, verbose=0)


if __name__ == '__main__':
    unittest.main()
