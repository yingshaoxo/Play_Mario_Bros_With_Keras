from model import generate_complex_model
import tensorflow as tf
import os
import numpy as np

from config import HISTORY_LENGTH

import time
import random
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env,  SIMPLE_MOVEMENT)
env.reset()

model_file_path = './nn_model.HDF5'
final_model_file_path = './final_nn_model.HDF5'
if os.path.exists(model_file_path):
    model = tf.keras.models.load_model(model_file_path)
else:
    model = generate_complex_model()


identity = np.identity(env.action_space.n)  # for quickly get a hot vector, like 0001000000000000


def one_element_of_history_array_to_vector(one_element_of_history_array):
    vector = []
    for item in one_element_of_history_array:
        value = item["value"]
        times = item["times"]
        vector.append(value)
        vector.append(times)
    return vector


def train_once(history_x_position, history_y_position, history_action, image, action):
    global model

    history_x_position = one_element_of_history_array_to_vector(history_x_position)
    history_y_position = one_element_of_history_array_to_vector(history_y_position)
    history_action = one_element_of_history_array_to_vector(history_action)

    action = identity[action: action+1]

    model.train_on_batch(
        x={
            'history_x_position': np.expand_dims(np.array(history_x_position), axis=0),
            'history_y_position': np.expand_dims(np.array(history_y_position), axis=0),
            'history_action': np.expand_dims(np.array(history_action), axis=0),
            'image': np.expand_dims(image, axis=0),
        },
        y=action,
    )


def predict_once(history_x_position, history_y_position, history_action, image):
    global model

    history_x_position = one_element_of_history_array_to_vector(history_x_position)
    history_y_position = one_element_of_history_array_to_vector(history_y_position)
    history_action = one_element_of_history_array_to_vector(history_action)

    result = model.predict({
        'history_x_position': np.expand_dims(np.array(history_x_position), axis=0),
        'history_y_position': np.expand_dims(np.array(history_y_position), axis=0),
        'history_action': np.expand_dims(np.array(history_action), axis=0),
        'image': np.expand_dims(image, axis=0),
    })

    result = np.argmax(result)
    print(SIMPLE_MOVEMENT[result])
    return result


identity = np.identity(env.action_space.n)  # for quickly get a hot vector, like 0001000000000000

image = None
reward = 0
done = None
info = None
action = 0

last_image = None

random_ration = 0.1

history_x_position = []
history_y_position = []
history_action = []
history_image = []

while 1:
    #####################
    # FIND A WAY
    ####################
    if random_ration > random.random():
        # take action randomly
        action = env.action_space.sample()
    else:
        # take action based on prediction
        if isinstance(image, (np.ndarray, np.generic)):
            if len(history_x_position) >= HISTORY_LENGTH and len(history_y_position) >= HISTORY_LENGTH and len(history_action) >= HISTORY_LENGTH:
                action = predict_once(history_x_position, history_y_position, history_action, image)

    #####################
    # TAKE ACTION
    ####################
    last_image = image
    image, reward, done, info = env.step(action)
    env.render()

    # update history data
    if len(history_action) == 0:
        history_action.append({"value": action, "times": 0})
    else:
        if action == history_action[-1]['value']:
            history_action[-1]["times"] += 1
        else:
            history_action.append({"value": action, "times": 0})
        history_action = history_action[-HISTORY_LENGTH:]

    if len(history_x_position) == 0:
        history_x_position.append({"value": info['x_pos'], "times": 0})
    else:
        if info['x_pos'] == history_x_position[-1]['value']:
            history_x_position[-1]["times"] += 1
        else:
            history_x_position.append({"value": info["x_pos"], "times": 0})
        history_x_position = history_x_position[-HISTORY_LENGTH:]

    if len(history_y_position) == 0:
        history_y_position.append({"value": info['y_pos'], "times": 0})
    else:
        if info['y_pos'] == history_y_position[-1]['value']:
            history_y_position[-1]["times"] += 1
        else:
            history_y_position.append({"value": info["y_pos"], "times": 0})
        history_y_position = history_y_position[-HISTORY_LENGTH:]

    if len(history_image) == 0:
        history_image.append({"value": image.copy(), "times": 0})
    else:
        if np.array_equal(image.copy(), history_image[-1]['value']):
            history_image[-1]["times"] += 1
        else:
            history_image.append({"value": image.copy(), "times": 0})
        history_image = history_image[-HISTORY_LENGTH:]

    if done or reward < -5:
        # you died, sorry
        env.reset()

        history_x_position = []
        history_y_position = []
        history_action = []
        history_image = []

    #####################
    # THINK ABOUT IT
    ####################
    #if reward != 0:
    #    if isinstance(last_image, (np.ndarray, np.generic)):
    #        if len(history_x_position) >= HISTORY_LENGTH and len(history_y_position) >= HISTORY_LENGTH and len(history_action) >= HISTORY_LENGTH:
    #            train_once(history_x_position, history_y_position, history_action, image, action)

    #####################
    # SAVE PROGRESS
    ####################
    # if info['x_pos'] > max_x_pos:
    #    differ = abs(info['x_pos'] - max_x_pos)
    #    if differ > 200:
    #        max_x_pos = info['x_pos']
    #        io.write_settings("max_x_pos", int(max_x_pos))
    #        if info['life'] == 2:
    #            model.save(model_file_path)

    # if info["stage"] == 2:
    #    model.save(final_model_file_path)
    #    input("congraduations!")
    #    exit()
