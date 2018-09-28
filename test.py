#!/usr/bin/env python3

import gym
import requests
import json

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)

ob = env.reset()
total_reward = 0
while True:
    r = requests.post('http://localhost:5000/forward', json={"state": ob.tolist()})
    r_obj = r.json()
    ob, reward, done, info = env.step(r_obj['action'])
    requests.post('http://localhost:5000/backward', json={"id": r_obj['id'], "reward": reward, "done": done})

    total_reward += reward

    if done:
        break

print(total_reward)
