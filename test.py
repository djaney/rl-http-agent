#!/usr/bin/env python3

import gym
import requests

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)

for sess_id in range(1000):
    ob = env.reset()
    total_reward = 0
    while True:
        r = requests.post('http://localhost:5000/forward', json={"state": ob.tolist()})
        r_obj = r.json()
        ob, reward, done, info = env.step(r_obj['action'])
        requests.post('http://localhost:5000/backward', json={"id": r_obj['id'], "reward": reward, "done": done, "session_id": str(sess_id)})

        total_reward += reward

        if done:
            break
    print("Score: {} ".format(total_reward))
