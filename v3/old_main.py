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

max_x_pos = io.read_settings("max_x_pos", 0)
history_x_pos_list_where_I_die = []
learning_area = 30
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

        if randint(1, 10) == 1 or not isinstance(last_state, (np.ndarray, np.generic)):
            action = env.action_space.sample()
            print(f"                            |  random  |  ")
        else:
            action = np.argmax(model.predict(np.expand_dims(last_state, axis=0)))
            print(f"                            |  predict  |  ")
            print(f"x_pos: {info['x_pos']}  |  reward: {reward}")

        if isinstance(state, (np.ndarray, np.generic)) and info != None:
            last_state = state
            last_info = info
        state, reward, done, info = env.step(action)
        print(f"x_pos: {info['x_pos']}  |  reward: {reward}")
        if isinstance(last_state, (np.ndarray, np.generic)) and last_info != None:
            if ((reward > 0) and (len(history_x_pos_list_where_I_die) > 0) and (history_x_pos_list_where_I_die[0]-learning_area < last_info['x_pos'] < history_x_pos_list_where_I_die[0]+learning_area)):
                print(f"curent: {last_info['x_pos']}, historical: {history_x_pos_list_where_I_die[0]}")
                if last_info["x_pos"] > history_x_pos_list_where_I_die[0]:
                    history_x_pos_list_where_I_die.pop(0)
                    print("fucked")
                    time.sleep(1)

                model.train_on_batch(x=np.expand_dims(last_state, axis=0), y=identity[action: action+1])
                print(f"                                            training happend")

        env.render()

        # additional operation
        if info['x_pos'] > max_x_pos:
            max_x_pos = info['x_pos']
            io.write_settings("max_x_pos", int(max_x_pos))
            if info['life'] == 2:
                model.save(model_file_path)
                pass

        if info["stage"] == 2:
            model.save(final_model_file_path)
            input("congraduations!")
            exit()

env.close()
