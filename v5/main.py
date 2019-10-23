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

model_file_path = './nn_model.HDF5'
final_model_file_path = './final_nn_model.HDF5'
if os.path.exists(model_file_path):
    model = tf.keras.models.load_model(model_file_path)
else:
    model = generate_complex_model()

# env.action_space.sample() = numbers, for example, 0,1,2,3...
# state = RGB of raw picture; is a numpy array with shape (240, 256, 3)
# reward = int; for example, 0, 1 ,2, ...
# done = False or True
# info = {'coins': 0, 'flag_get': False, 'life': 3, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40}

identity = np.identity(env.action_space.n)  # for quickly get a hot vector, like 0001000000000000

state = None
reward = 0
done = True
info = None

last_state = None
last_info = None

max_x_pos = io.read_settings("max_x_pos", 0)  # save personal record, so we can save the best model when we hit our own limitation
history_x_pos_list_where_I_die = [0]  # define those points where I failed
learning_area = 50  # how wide the death area is. we will learn from past failures
history_data = []  # for traning when passing died point
last_place_I_die = 0
longest_history_data = []

history_actions = []
history_x_pos = []
history_y_pos = []

training_couting = io.read_settings("training_couting", 1)
random_ration = 0.1

history_rewards = []
while 1:
    for step in range(1000):
        random_ration = 0.5/training_couting

        if done or reward < -5:
            state = env.reset()
            if last_info != None:
                if len(history_x_pos_list_where_I_die) > 0:
                    if last_info['x_pos'] < history_x_pos_list_where_I_die[0]:
                        history_x_pos_list_where_I_die.insert(0, last_info['x_pos'])
                else:
                    history_x_pos_list_where_I_die.insert(0, last_info['x_pos'])
            if len(history_data) > len(longest_history_data):
                longest_history_data = history_data.copy()
            history_data = []  # we don't want to learn anything that leading us to die
            history_rewards = []

        if (random_ration > random.random()) or not isinstance(last_state, (np.ndarray, np.generic)):
            # if (random.randint(1, 10) == 1) or not isinstance(last_state, (np.ndarray, np.generic)):
            action = env.action_space.sample()
            print(f"                            |  random  |  ")
        else:
            if len(history_actions) == 32 and len(history_x_pos) == 32:
                action = np.argmax(model.predict({
                    'img': np.expand_dims(last_state, axis=0),
                    'action': np.expand_dims(np.array(history_actions), axis=0),
                    'x_position': np.expand_dims(np.array(history_x_pos), axis=0),
                    'y_position': np.expand_dims(np.array(history_y_pos), axis=0),
                }))
                print(f"                            |  predict  |  ")
                print(f"x_pos: {info['x_pos']}  |  reward: {reward}")

        #if len(history_rewards) > 0 and (list(map(lambda x: x == 0 or x == -1, history_rewards)).count(True))/len(history_rewards) > 0.8:
        #    action = 3

        if isinstance(state, (np.ndarray, np.generic)) and info != None:
            last_state = state
            last_info = info
        #print(f"action: {action}")
        state, reward, done, info = env.step(action)

        history_actions.append(action)
        history_actions = history_actions[-32:]
        history_x_pos.append(info['x_pos'])
        history_x_pos = history_x_pos[-32:]
        history_y_pos.append(info['y_pos'])
        history_y_pos = history_y_pos[-32:]

        history_rewards.append(reward)

        print(f"x_pos: {info['x_pos']}  |  reward: {reward}")
        if isinstance(last_state, (np.ndarray, np.generic)) and last_info != None:
            if (len(history_x_pos_list_where_I_die) > 0):
                if last_place_I_die > history_x_pos_list_where_I_die[0]:
                    last_place_I_die = 0
                print(f"                                           last_place: {last_place_I_die}, curent: {last_info['x_pos']}, historical: {history_x_pos_list_where_I_die[0]}")
            if ((reward > 0 or info['y_pos'] > 0) and (len(history_x_pos_list_where_I_die) > 0) and (last_place_I_die < last_info['x_pos'] < history_x_pos_list_where_I_die[0]+learning_area*1.5)):
                if len(history_actions) == 32 and len(history_x_pos) == 32:
                    history_data.append({
                        "state": last_state,
                        "action": action,
                        "history_actions": history_actions.copy(),
                        "history_x_pos": history_x_pos.copy(),
                        "history_y_pos": history_y_pos.copy(),
                    })
                if last_info["x_pos"] > history_x_pos_list_where_I_die[0]+learning_area:
                    place = history_x_pos_list_where_I_die.pop(0)
                    if len(history_x_pos_list_where_I_die) > 0:
                        if place < history_x_pos_list_where_I_die[0]:
                            last_place_I_die = place
                    print(f"                                            learning happend: {len(history_data)}")
                    training_couting += 1
                    io.write_settings("training_couting", int(training_couting))
                    for moment in history_data+longest_history_data:
                        model.train_on_batch(
                            x={
                                'img': np.expand_dims(moment["state"], axis=0),
                                'action': np.expand_dims(np.array(moment["history_actions"]), axis=0),
                                'x_position': np.expand_dims(np.array(moment["history_x_pos"]), axis=0),
                                'y_position': np.expand_dims(np.array(moment["history_y_pos"]), axis=0),
                            },
                            y=identity[moment['action']: moment['action']+1]
                        )

                    if len(history_data) > len(longest_history_data):
                        longest_history_data = history_data.copy()
                    history_data = []
                    history_rewards = []

        env.render()

        # additional operation
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

env.close()
