from model import generate_complex_model
import tensorflow as tf
import os
import numpy as np

import time
import random
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from auto_everything.base import IO
io = IO()

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env,  SIMPLE_MOVEMENT)
env.reset()

model_file_path = './nn_model.HDF5'
final_model_file_path = './final_nn_model.HDF5'
if os.path.exists(model_file_path):
    model = tf.keras.models.load_model(model_file_path)
else:
    model = generate_complex_model()


def train_once(last_state, history_actions, history_x_pos, history_y_pos, action):
    global model
    model.train_on_batch(
        x={
            'img': np.expand_dims(last_state, axis=0),
            'action': np.expand_dims(np.array(history_actions), axis=0),
            'x_position': np.expand_dims(np.array(history_x_pos), axis=0),
            'y_position': np.expand_dims(np.array(history_y_pos), axis=0),
        },
        y=identity[action: action+1]
    )


identity = np.identity(env.action_space.n)  # for quickly get a hot vector, like 0001000000000000

action = 0

state = None
reward = 0
done = None
info = None

last_state = None
last_info = None

max_x_pos = io.read_settings("max_x_pos", 0)  # save personal record, so we can save the best model when we hit our own limitation

history_actions = []
history_x_pos = []
history_y_pos = []

training_couting = io.read_settings("training_couting", 1)
random_ration = 0.1

history_rewards = []
while 1:
    #random_ration = 1000/training_couting

    #####################
    # ACTION TAKING
    ####################
    if random_ration > random.random():  # exploration
        action = env.action_space.sample()
        print(f"                            |  random  |  ")
    else:  # prediction by experience
        if isinstance(state, (np.ndarray, np.generic)):
            if len(history_actions) >= 101 and len(history_x_pos) >= 101 and len(history_y_pos) >= 101:
                action = np.argmax(model.predict({
                    'img': np.expand_dims(state, axis=0),
                    'action': np.expand_dims(np.array(history_actions[-100:]), axis=0),
                    'x_position': np.expand_dims(np.array(history_x_pos[-100:]), axis=0),
                    'y_position': np.expand_dims(np.array(history_y_pos[-100:]), axis=0),
                }))
                print(f"                            |  predict  |  ")
                print(f"x_pos: {info['x_pos']}  |  reward: {reward}")

    if len(history_rewards) > 0 and (list(map(lambda x: x == 0 or x == -1, history_rewards)).count(True))/len(history_rewards) > 0.8:  # jump when stuck at the same place too long
        if isinstance(state, (np.ndarray, np.generic)) and info != None:
            last_state = state
            last_info = info

        action = 3
        state, reward, done, info = env.step(action)

        history_actions.append(action)
        history_actions = history_actions[-200:]
        history_x_pos.append(info['x_pos'])
        history_x_pos = history_x_pos[-200:]
        history_y_pos.append(info['y_pos'])
        history_y_pos = history_y_pos[-200:]

        if isinstance(last_state, (np.ndarray, np.generic)):
            if len(history_actions) >= 101 and len(history_x_pos) >= 101 and len(history_y_pos) >= 101:
                training_couting += 1
                io.write_settings("training_couting", int(training_couting))

                print(f"                                            learning happend with action: {SIMPLE_MOVEMENT[action]}")
                train_once(last_state, history_actions[-101:-1], history_x_pos[-101:-1], history_y_pos[-101:-1], action)

        temp = 15
        while 1:
            if isinstance(state, (np.ndarray, np.generic)) and info != None:
                last_state = state
                last_info = info

            action = 2
            state, reward, done, info = env.step(action)

            history_actions.append(action)
            history_actions = history_actions[-200:]
            history_x_pos.append(info['x_pos'])
            history_x_pos = history_x_pos[-200:]
            history_y_pos.append(info['y_pos'])
            history_y_pos = history_y_pos[-200:]

            if isinstance(last_state, (np.ndarray, np.generic)):
                if len(history_actions) >= 101 and len(history_x_pos) >= 101 and len(history_y_pos) >= 101:
                    training_couting += 1
                    io.write_settings("training_couting", int(training_couting))

                    print(f"                                            learning happend with action: {SIMPLE_MOVEMENT[action]}")
                    train_once(last_state, history_actions[-101:-1], history_x_pos[-101:-1], history_y_pos[-101:-1], action)

            if done:
                env.reset()
            env.render()
            temp -= 1
            if temp < 0:
                break

        if (len(history_rewards)) > 300:
                history_rewards = []

    #####################
    # Do it!
    ####################
    if isinstance(state, (np.ndarray, np.generic)) and info != None:
        last_state = state
        last_info = info

    state, reward, done, info = env.step(action)
    env.render()
    print(f"x_pos: {info['x_pos']}  |  reward: {reward}")

    history_actions.append(action)
    history_actions = history_actions[-200:]
    history_x_pos.append(info['x_pos'])
    history_x_pos = history_x_pos[-200:]
    history_y_pos.append(info['y_pos'])
    history_y_pos = history_y_pos[-200:]

    history_rewards.append(reward)

    if done or reward < -5:  # you died, sorry
        state = env.reset()
        last_state = None
        last_info = None

        history_actions = []
        history_x_pos = []
        history_y_pos = []

        history_rewards = []

    #####################
    # Think about it!
    ####################
    if (reward > 0):
        if isinstance(last_state, (np.ndarray, np.generic)):
            if len(history_actions) >= 101 and len(history_x_pos) >= 101 and len(history_y_pos) >= 101:
                training_couting += 1
                io.write_settings("training_couting", int(training_couting))

                print(f"                                            learning happend with action: {SIMPLE_MOVEMENT[action]}")
                train_once(last_state, history_actions[-101:-1], history_x_pos[-101:-1], history_y_pos[-101:-1], action)

    #####################
    # Save progress
    ####################
    if info['x_pos'] > max_x_pos:
        differ = abs(info['x_pos'] - max_x_pos)
        if differ > 200:
            max_x_pos = info['x_pos']
            io.write_settings("max_x_pos", int(max_x_pos))
            if info['life'] == 2:
                model.save(model_file_path)

    if info["stage"] == 2:
        model.save(final_model_file_path)
        input("congraduations!")
        exit()
