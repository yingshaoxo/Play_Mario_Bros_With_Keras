from model import generate_model
import tensorflow as tf
import os
import numpy as np
from random import randint
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
import random
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
last_action = None
identity = np.identity(env.action_space.n)  # for quickly get a hot vector, like 0001000000000000

x_pos = 0
max_x_pos = io.read_settings("max_x_pos", 0)
perfect_model = model
reward = 0
failer_mode = False
max_attemps_in_failer_mode = 50
history_rewards = []
while 1:
    model = perfect_model

    for step in range(1000):
        if done:
            state = env.reset()
            model = perfect_model

        if reward < 0:
            ratio = 1
        else:
            ratio = abs(reward)
        if (random.randint(1, max(4, len(history_rewards)*ratio)) == 1 and failer_mode == True) or not isinstance(last_state, (np.ndarray, np.generic)):
            action = env.action_space.sample()
            print("random              ")
        else:
            action = np.argmax(model.predict(np.expand_dims(last_state, axis=0)))
            print("            predict")

        state, reward, done, info = env.step(action)
        if reward <= 0:
            history_rewards = []
            if failer_mode == False:
                failer_mode = True

        last_state = state
        last_action = action
        if reward >= 0 and x_pos > max_x_pos:
            history_rewards.append(reward)
            model.train_on_batch(x=np.expand_dims(last_state, axis=0), y=identity[action: action+1])
            print(f"                        training happens: {reward}\nx_pos:{info['x_pos']}")
        elif reward < 0:
            action = env.action_space.sample()
            model.train_on_batch(x=np.expand_dims(last_state, axis=0), y=identity[action: action+1])
            print(f"                        training happens: {reward}\nx_pos:{info['x_pos']}")

        env.render()

        x_pos = info["x_pos"]
        if x_pos > max_x_pos:
            max_x_pos = x_pos
            io.write_settings("max_x_pos", int(max_x_pos))
            if info["life"] == 2:
                failer_mode = False
                perfect_model = model
                model.save(model_file_path)
        if info["stage"] == 2:
            io.write_settings("max_x_pos", int(max_x_pos))
            model.save(final_model_file_path)
            input("congraduations!")
            exit()

env.close()
