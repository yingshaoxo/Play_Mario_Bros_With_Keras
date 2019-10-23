from model import generate_model
import tensorflow as tf
import os
import numpy as np

import time
from random import randint
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
    model = generate_model()

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
while 1:
    for step in range(1000):
        if done or reward < -5:
            state = env.reset()
            if last_info != None:
                if len(history_x_pos_list_where_I_die) > 0:
                    if last_info['x_pos'] < history_x_pos_list_where_I_die[0]:
                        history_x_pos_list_where_I_die.insert(0, last_info['x_pos'])
                else:
                    history_x_pos_list_where_I_die.insert(0, last_info['x_pos'])
            history_data = []  # we don't want to learn anything that leading us to die

        if reward == -1 or (randint(1, 10) == 1) or not isinstance(last_state, (np.ndarray, np.generic)):
            action = env.action_space.sample()
            print(f"                            |  random  |  ")
        else:
            action = np.argmax(model.predict(np.expand_dims(last_state, axis=0)))
            print(f"                            |  predict  |  ")
            print(f"x_pos: {info['x_pos']}  |  reward: {reward}")

        if isinstance(state, (np.ndarray, np.generic)) and info != None:
            last_state = state
            last_info = info
        #print(f"action: {action}")
        state, reward, done, info = env.step(action)
        print(f"x_pos: {info['x_pos']}  |  reward: {reward}")
        if isinstance(last_state, (np.ndarray, np.generic)) and last_info != None:
            if (len(history_x_pos_list_where_I_die) > 0):
                if last_place_I_die > history_x_pos_list_where_I_die[0]:
                    last_place_I_die = 0
                print(f"                                           last_place: {last_place_I_die}, curent: {last_info['x_pos']}, historical: {history_x_pos_list_where_I_die[0]}")
            if ((reward > 0) and (len(history_x_pos_list_where_I_die) > 0) and (last_place_I_die < last_info['x_pos'] < history_x_pos_list_where_I_die[0]+learning_area*1.5)):
                history_data.append({
                    "state": last_state,
                    "action": action
                })
                if last_info["x_pos"] > history_x_pos_list_where_I_die[0]+learning_area:
                    place = history_x_pos_list_where_I_die.pop(0)
                    if len(history_x_pos_list_where_I_die) > 0:
                        if place < history_x_pos_list_where_I_die[0]:
                            last_place_I_die = place
                    print(f"                                            learning happend: {len(history_data)}")
                    for moment in history_data:
                        #model.train_on_batch(x=np.expand_dims(moment["state"], axis=0), y=identity[moment['action']: moment['action']+1])
                        pass

                    history_data = []

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
