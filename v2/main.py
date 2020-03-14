import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import random
from pprint import pprint
import time
from config import HISTORY_LENGTH  # it's a local py file: config.py
from config import MY_MOVEMENT  # it's a local py file: config.py
from model import generate_model

import tensorflow as tf
import os
import numpy as np


env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env,  MY_MOVEMENT)
env.reset()


model_file_path = './nn_model'
final_model_file_path = './final_nn_model'
if os.path.exists(model_file_path):
    model = tf.keras.models.load_model(model_file_path)
else:
    model = generate_model()


identity = np.identity(env.action_space.n)  # for quickly get a hot vector, like 0001000000000000

image = None
reward = 0
done = None
info = None
action = 0

last_image = None
max_distance = 0

random_ration = 0.1  # use it when doing RL

history_inputs = []


def get_random_action():
    action = env.action_space.sample()
    print(" "*15 + "+".join(MY_MOVEMENT[action]))
    return action


def predict_once(image):
    global model

    result = model.predict(np.expand_dims(np.array(image), axis=0))

    action = np.argmax(result)
    print("+".join(MY_MOVEMENT[action]))
    return action


def train_once(image, action):
    global model

    print(" "*30 + "+".join(MY_MOVEMENT[action]))

    action = identity[action: action+1]

    model.train_on_batch(
        x=np.expand_dims(np.array(image), axis=0),
        y=action,
    )


def train_array():
    global history_inputs

    print(" "*30 + "Training...")

    for (image, action) in history_inputs:
        train_once(image, action)

    history_inputs = []


x_list = []
while 1:
    #####################
    # FIND A WAY
    ####################
    if random_ration > random.random():
        # take action randomly
        action = get_random_action()
    else:
        # try to take action based on prediction
        if isinstance(image, (np.ndarray, np.generic)):
            # take action based on prediction
            if isinstance(last_image, (np.ndarray, np.generic)):
                #history_inputs.append((last_image, action))
                pass
            action = predict_once(image)
        else:
            # take action randomly
            action = get_random_action()

    #####################
    # TAKE ACTION
    ####################
    last_image = image
    for _ in range(20):
        image, reward, done, info = env.step(action)
        env.render()

        if done or reward < -5:
            # you died, sorry
            env.reset()
            max_distance = 0
            x_list = []
            history_inputs = []

    #####################
    # THINK ABOUT IT
    ####################
    current_distance = info['x_pos']
    progress = current_distance - max_distance
    print(" "*50 + str(progress))
    if progress > 0:#HISTORY_LENGTH**3:
        max_distance = current_distance
        # We should think about a strategy to make sure we train it with right data
        # Like split history to levels: if we live, for the experience we have, the older, the better
        if isinstance(last_image, (np.ndarray, np.generic)):
            #train_array()
            train_once(last_image, action)

    #####################
    # SAVE PROGRESS
    ####################
    x_list.append(info['x_pos'])
    x_list = x_list[-30:]
    if len(x_list) >= 30:
        value = np.mean(x_list) - x_list[-1]
        print(" "*60 + str(value))
        if abs(value) < 1:
            model.save(model_file_path)
            env.reset()
            #max_distance = 0
            x_list = []
            history_inputs = []

    if info["stage"] == 2:
        model.save(final_model_file_path)
        input("congraduations!")
        exit()
