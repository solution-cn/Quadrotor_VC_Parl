# -*- coding = utf-8 -*-
# @Time : 2020/6/26 1:21
# @Author : solution
# @File : velocity.py
# @Software: PyCharm

import numpy as np
from parl.utils import action_mapping
from rlschool import make_env

def evaluate_episode(env,render=False):
    env_reward = []
    for j in range(5):
        env.reset()
        d_r=0
        while True:
            actuall = np.array([-1,-1,-1,-1],dtype='float32')
            actuall = action_mapping(actuall, env.action_space.low[0], env.action_space.high[0])
            next_obs, reward, done, info = env.step(actuall)
            d_r += reward
            if render:
                env.render()
            if done:
                break
        env_reward.append(d_r)
    env_reward.append(np.mean(env_reward))
    return env_reward


env = make_env("Quadrotor", task="velocity_control")
Test_Reward = evaluate_episode(env,render=True)
print(Test_Reward)