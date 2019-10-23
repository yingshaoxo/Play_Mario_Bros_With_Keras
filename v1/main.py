from model import generate_model
import tensorflow as tf
import os
import numpy as np
from random import randint
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
#from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
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

done = True
last_state = None
identity = np.identity(env.action_space.n)  # for quickly get a hot vector, like 0001000000000000

x_pos = 0
max_x_pos = io.read_settings("max_x_pos", 0)
life = 2
perfect_model = model
while 1:
    model = perfect_model

    for step in range(1000):
        if done:
            state = env.reset()

        if randint(1, 10) == 1 or not isinstance(last_state, (np.ndarray, np.generic)):
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.expand_dims(last_state, axis=0)))
            # print(action)

        state, reward, done, info = env.step(action)
        last_state = state
        if reward > 1 and x_pos > max_x_pos:
            model.train_on_batch(x=np.expand_dims(last_state, axis=0), y=identity[action: action+1])
            print(f"training happens: {reward}\nx_pos:{info['x_pos']}")

        env.render()

        x_pos = info["x_pos"]
        if x_pos > max_x_pos:
            max_x_pos = x_pos
            io.write_settings("max_x_pos", int(max_x_pos))
            if info["life"] == 2:
                perfect_model = model
                model.save(model_file_path)
        if info["stage"] == 2:
            io.write_settings("max_x_pos", int(max_x_pos))
            model.save(final_model_file_path)
            input("congraduations!")
            exit()

        life = info["life"]

env.close()
