#!/usr/bin/env python3

import gym
import requests

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)

for sess_id in range(1000):
    ob = env.reset()
    total_reward = 0
    counter = 0
    current_action = None
    while True:
        # repeat actions for 4 frames
        if counter >= 4 or current_action is None:
            r = requests.post('http://localhost:5000/forward', json={"state": ob.tolist()})
            r_obj = r.json()
            action = r_obj['action']

            ob, reward, done, info = env.step(action)
            requests.post('http://localhost:5000/backward',
                          json={"id": r_obj['id'], "reward": reward, "done": done, "session_id": str(sess_id)})
            current_action = action
            counter = 0
        else:
            action = current_action

            ob, reward, done, info = env.step(action)

            counter += 1
        env.render()


        total_reward += reward

        if done:
            break
    print("Score: {} ".format(total_reward))
