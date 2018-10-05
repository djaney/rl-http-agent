#!/usr/bin/env python3

import gym
import requests
from core.environments import STATUS_RESTING, STATUS_WAITING_FOR_RESET_OBSERVATIONS

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)

ob = env.reset()
total_reward = 0
while True:
    r = requests.get('http://localhost:5000/get_status')
    status = r.text.strip()
    print(status)
    if status == STATUS_RESTING:
        requests.get('http://localhost:5000/start')
    elif status == STATUS_WAITING_FOR_RESET_OBSERVATIONS:
        ob = env.reset()
        requests.post('http://localhost:5000/send_reset', json=ob.tolist())


