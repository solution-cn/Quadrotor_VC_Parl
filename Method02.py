#-*- coding = utf-8 -*-
#@Time : 2020/6/30 2:56
#@Author : solution
#@File : L3.py
#@Software: PyCharm

import os
import numpy as np
import parl
from paddle import fluid
from parl import layers
from parl.utils import logger
from parl.utils import action_mapping
from parl.utils import ReplayMemory
from rlschool import make_env

# 超参数
A_lr = 2e-5
C_lr = 5e-4

Gamma = 0.90
Tau = 0.001
Reward_Scale = 0.1
Test_Round = 1e4
Memory_Warm_Up = 1e4
Train_Step = 1e6
Max_Size = 1e6
Batch_Size = 256


class QModel(parl.Model):
    def __init__(self, act_dim):
        self.ActorModel = ActorModel(act_dim)
        self.CriticModel = CriticModel()

    def policy(self, obs):
        return self.ActorModel.policy(obs)

    def value(self, obs, act):
        return self.CriticModel.value(obs, act)

    def get_actor_params(self):
        return self.ActorModel.parameters()


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        self.fc1 = layers.fc(size=32, act='relu')
        self.fc2 = layers.fc(size=act_dim, act='tanh')

    def policy(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        return out


class CriticModel(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(size=64, act='relu')
        self.fc2 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        temp = layers.concat([obs, act], axis=1)
        Q = self.fc1(temp)
        Q = self.fc2(Q)
        Q = layers.squeeze(Q, axes=[1])
        return Q


from parl.algorithms import DDPG


class QAgent(parl.Agent):
    def __init__(self, alg, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(QAgent, self).__init__(alg)
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.predict_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.predict_program):
            obs = layers.data(name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(name='next_obs', shape=[self.obs_dim], dtype='float32')
            done = layers.data(name='done', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs, done)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(self.predict_program, feed={'obs': obs}, fetch_list=[self.pred_act])[0]
        return act

    def learn(self, obs, act, reward, next_obs, done):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        }
        C_cost = self.fluid_executor.run(self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return C_cost


def run_episode(env, agent, rpm,render=False):
    step = 0
    total_reward = 0
    obs = env.reset()
    while True:
        step += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs)
        action = np.squeeze(action)
        action = np.random.normal(action, 1.0)
        action = np.clip(action, -1.0, 1.0)
        actuall = action
        actuall = action_mapping(actuall, env.action_space.low[0], env.action_space.high[0])
        next_obs, reward, done, info = env.step(actuall)
        vx_1=abs(info['b_v_x']-info['next_target_g_v_x'])
        vy_1=abs(info['b_v_x']-info['next_target_g_v_y'])
        vz_1=abs(info['b_v_x']-info['next_target_g_v_z'])
        vx_2=pow(info['b_v_x']-info['next_target_g_v_x'],2)
        vy_2=pow(info['b_v_y']-info['next_target_g_v_y'],2)
        vz_2=pow(info['b_v_z']-info['next_target_g_v_z'],2)
        reward_adept=-0.1*(vx_1+vy_1+vz_1+vx_2+vy_2+vz_2)
        rpm.append(obs, action, Reward_Scale * reward_adept , next_obs,done)
        if rpm.size() > Memory_Warm_Up:
            batch_obs, batch_act, batch_reward, batch_next_obs, batch_done = rpm.sample_batch(Batch_Size)
            C_cost = agent.learn(batch_obs, batch_act, batch_reward, batch_next_obs, batch_done)
        obs = next_obs
        total_reward += reward_adept
        if render:
            env.render()
        if done:
            break
    return step, total_reward


def evaluate_episode(env, agent,render=False):
    total_reward = []
    env_reward = []
    for j in range(5):
        obs = env.reset()
        c_r=0
        d_r=0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs)
            action = np.squeeze(action)
            action = np.clip(action, -1.0, 1.0)
            actuall = action
            actuall = action_mapping(actuall, env.action_space.low[0], env.action_space.high[0])
            next_obs, reward, done, info = env.step(actuall)
            obs = next_obs
            vx_1=abs(info['b_v_x']-info['next_target_g_v_x'])
            vy_1=abs(info['b_v_x']-info['next_target_g_v_y'])
            vz_1=abs(info['b_v_x']-info['next_target_g_v_z'])
            vx_2=pow(info['b_v_x']-info['next_target_g_v_x'],2)
            vy_2=pow(info['b_v_y']-info['next_target_g_v_y'],2)
            vz_2=pow(info['b_v_z']-info['next_target_g_v_z'],2)
            reward_adept=-0.1*(vx_1+vy_1+vz_1+vx_2+vy_2+vz_2)
            d_r += reward
            c_r += reward_adept
            if render:
                env.render()
            if done:
                break
        total_reward.append(c_r)
        env_reward.append(d_r)
    total_reward.append(np.mean(total_reward))
    env_reward.append(np.mean(env_reward))
    return total_reward,env_reward


env = make_env("Quadrotor", task="velocity_control")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
logger.info('obs_dim:{} , act_dim:{}'.format(obs_dim, act_dim))

model = QModel(act_dim)
alg = DDPG(model, gamma=Gamma, tau=Tau, actor_lr=A_lr, critic_lr=C_lr)
agent = QAgent(alg, obs_dim, act_dim)

rpm = ReplayMemory(int(Max_Size), obs_dim, act_dim)

# if os.path.exists('./M2_-1094_Over.ckpt'):
#     agent.restore('./M2_-1094_Over.ckpt')
# else:
#     exit(1)

test_flag = 0
total_step = 0
reward_max = -1094
while True:
    step, reward = run_episode(env, agent, rpm)
    total_step += step
    logger.info('Step:{} , Train Reward:{}'.format(total_step, reward))

    if total_step // Test_Round == test_flag:
        Test_Reward,Env_Reward = evaluate_episode(env, agent)
        logger.info('C_R:{} , D_R:{}'.format(Test_Reward[5],Env_Reward[5]))
        if reward_max < Test_Reward[5]:
            reward_max = Test_Reward[5]
            model_path = 'model/R%.0f_Step%d.ckpt'%(Test_Reward[5],total_step)
            agent.save(model_path)
        if total_step >= Train_Step:
            model_path = 'model/R_%.0f_Over.ckpt' % (Test_Reward[5])
            agent.save(model_path)
            break
        test_flag += 1

# Test_Reward,Env_Reward = evaluate_episode(env, agent,render=True)
# print(Test_Reward,Env_Reward)