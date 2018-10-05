#!/usr/bin/env python3

import gym
import requests
from core.environments import STATUS_DONE, STATUS_WAITING_FOR_RESET_OBSERVATIONS, STATUS_WAITING_FOR_ACTION_RESULT

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)

ob = env.reset()
total_reward = 0
while True:
    r = requests.get('http://localhost:5000/get_status')
    status = r.text.strip()
    if status == STATUS_DONE:
        requests.get('http://localhost:5000/start')
    elif status == STATUS_WAITING_FOR_RESET_OBSERVATIONS:
        ob = env.reset()
        requests.post('http://localhost:5000/send_reset', json=ob.tolist())
    elif status == STATUS_WAITING_FOR_ACTION_RESULT:
        r = requests.get('http://localhost:5000/get_action')
        action = int(r.text.strip())
        ob, reward, done, info = env.step(action)
        total_reward += reward
        requests.post('http://localhost:5000/send_step_result', json=[ob.tolist(), reward, done, info])
        if done:
            print("Total Reward: ", total_reward)
            total_reward = 0
            ob = env.reset()
            requests.post('http://localhost:5000/send_reset', json=ob.tolist())

    env.render()
