# coding=utf-8

"""
Policy Gradient, Reinforcement Learning.
The cart pole example
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt
from RL_brain_Dueling import DuelingDQN
import gym_super_mario_bros
import numpy as np
import cv2
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY, GO_GO_GO, GO_GO

""" ===================================================================== """
save_path = 'saved_networks_pixle'
n_actions = 7
n_features = 6400
CONTINUE = True

# env = gym_super_mario_bros.make('SuperMarioBrosNoFrameskip-v0')
env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

MEMORY_SIZE = 3000
ACTION_SPACE = n_actions
TOTAL_EPISODE = 1000000
FRAME_PER_ACTION = 1  # 12

max_accumulative_r = -99999
max_fitness = -99999

# RL = DuelingDQN(
#     n_actions=n_actions,
#     n_features=n_features,
#     learning_rate=0.02,
#     reward_decay=0.99,
#     e_greedy_increment=0.001
# )

RL = PolicyGradient(
    n_actions=n_actions,
    n_features=n_features,
    learning_rate=0.01,
    reward_decay=0.95,
    # output_graph=True,
)

if CONTINUE:
    print('Continue!')
    RL.load(save_path)

# global ACTION_SPACE
acc_r = [0]
total_steps = 0
total_score_this_save = 0
max_real_score_this_save = 0
for episode in range(TOTAL_EPISODE):
    if episode % 50 == 0 and episode != 0:
        RL.save(episode, save_path)
        print('==========  Save!  ==========')
        print('Save Path: ', save_path)
        print('episode: ', episode)
        print('total_score_this_save: ', total_score_this_save)
        print('max_real_score_this_save: ', max_real_score_this_save)
        total_score_this_save = 0
        max_real_score_this_save = 0
        print('')

    # initial observation
    # do_nothing = np.zeros(ACTION_SPACE)
    observation = env.reset()
    x_t = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = x_t.flatten()

    accumulative_r = 0.
    nop_count = 0
    pre_x = -1

    while True:

        # action_ = np.zeros(ACTION_SPACE)  # game action parameter
        action = 0  # 如果不符合 FRAME_PER_ACTION 条件 则不动
        if total_steps % FRAME_PER_ACTION == 0:
            # RL choose action based on observation
            action = RL.choose_action(s_t)  # (observation)
        else:
            action = 0  # do nothing

        observation_, reward, done, info = env.step(action)
        x_pos = info['x_pos']
        # env.render()  # 渲染

        # if x_pos == pre_x:
        #     nop_count += 1
        #     if nop_count > 2000:
        #         done = True
        #         nop_count = 0
        #         reward -= 500

        x_t1 = cv2.cvtColor(cv2.resize(observation_, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = x_t1.flatten()

        accumulative_r += reward

        # RL.store_transition(s_t, action, float(reward), s_t1)
        RL.store_transition(s_t, action, float(reward))

        if done:  #  or info['life'] < 3
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            # print("episode:", episode, "  reward:", int(running_reward))

            vt = RL.learn()

            if max_accumulative_r == -99999:
                max_accumulative_r = accumulative_r
            max_accumulative_r = accumulative_r > max_accumulative_r and accumulative_r or max_accumulative_r
            max_real_score_this_save = accumulative_r > max_real_score_this_save and accumulative_r or max_real_score_this_save

            total_score_this_save += accumulative_r
            print(episode % 50, ': accumulative_r / max_accumulative_r: %f / %f' % (accumulative_r, max_accumulative_r))

            # if episode == 0:
            #     plt.plot(vt)    # plot the episode vt
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()
            env.reset()
            break

        pre_x = x_pos
        s_t = s_t1
        total_steps += 1

































































