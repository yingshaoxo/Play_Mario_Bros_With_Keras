from config import HISTORY_LENGTH, HISTORY_LEVEL
from model import generate_complex_model


import tensorflow as tf
import os
import numpy as np


import time
from pprint import pprint
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

image = None
reward = 0
done = None
info = None
action = 0

last_image = None

initial_flag = True
#random_ration = 0.1 # use it when doing RL
random_ration = 0.0 # you can use this to see how well the supervised model does

history_x_position = []
history_y_position = []
history_action = []
history_image = []

history_x_position_array = []
history_y_position_array = []
history_action_array = []
history_image_array = []
image_array = []
action_array = []


def one_element_of_history_array_to_vector(one_element_of_history_array, only_value=False):
    try:
        one_element_of_history_array = one_element_of_history_array[:HISTORY_LENGTH]
        vector = []
        for item in one_element_of_history_array:
            value = item["value"]
            times = item["times"]
            vector.append(value)
            if only_value == False:
                vector.append(times)
        return vector
    except Exception as e:
        print(e)
        pprint(one_element_of_history_array)
        raise Exception("Fail to parse that array.")


def predict_once(history_x_position, history_y_position, history_action, history_image, image):
    global model

    history_x_position = one_element_of_history_array_to_vector(history_x_position)
    history_y_position = one_element_of_history_array_to_vector(history_y_position)
    history_action = one_element_of_history_array_to_vector(history_action)
    history_image = one_element_of_history_array_to_vector(history_image, only_value=True)

    result = model.predict({
        'history_x_position': np.expand_dims(np.array(history_x_position), axis=0),
        'history_y_position': np.expand_dims(np.array(history_y_position), axis=0),
        'history_action': np.expand_dims(np.array(history_action), axis=0),
        'history_image': np.expand_dims(np.array(history_image), axis=0),
        'image': np.expand_dims(image, axis=0),
    })

    result = np.argmax(result)
    print(SIMPLE_MOVEMENT[result])
    return result


def train_once(history_x_position, history_y_position, history_action, history_image, image, action):
    global model

    history_x_position = one_element_of_history_array_to_vector(history_x_position)
    history_y_position = one_element_of_history_array_to_vector(history_y_position)
    history_action = one_element_of_history_array_to_vector(history_action)
    history_image = one_element_of_history_array_to_vector(history_image, only_value=True)

    action = identity[action: action+1]

    model.train_on_batch(
        x={
            'history_x_position': np.expand_dims(np.array(history_x_position), axis=0),
            'history_y_position': np.expand_dims(np.array(history_y_position), axis=0),
            'history_action': np.expand_dims(np.array(history_action), axis=0),

            'history_image': np.expand_dims(np.array(history_image), axis=0),
            'image': np.expand_dims(image, axis=0),
        },
        y=action,
    )


def manage_history_array(how=None):
    global history_x_position_array, history_y_position_array, history_action_array, history_image_array, image_array, action_array

    if how == "update":
        history_x_position_array = history_x_position_array[-HISTORY_LENGTH*HISTORY_LEVEL:]
        history_y_position_array = history_y_position_array[-HISTORY_LENGTH*HISTORY_LEVEL:]
        history_action_array = history_action_array[-HISTORY_LENGTH*HISTORY_LEVEL:]
        history_image_array = history_image_array[-HISTORY_LENGTH*HISTORY_LEVEL:]
        image_array = image_array[-HISTORY_LENGTH*HISTORY_LEVEL:]
        action_array = action_array[-HISTORY_LENGTH*HISTORY_LEVEL:]
    elif how == "refresh":
        history_x_position_array = history_x_position_array[HISTORY_LENGTH:]
        history_y_position_array = history_y_position_array[HISTORY_LENGTH:]
        history_action_array = history_action_array[HISTORY_LENGTH:]
        history_image_array = history_image_array[HISTORY_LENGTH:]
        image_array = image_array[HISTORY_LENGTH:]
        action_array = action_array[HISTORY_LENGTH:]
    elif how == "reset":
        history_x_position_array = []
        history_y_position_array = []
        history_action_array = []
        history_image_array = []
        image_array = []
        action_array = []


def train_array(history_x_position_array, history_y_position_array, history_action_array, history_image_array, image_array, action_array):
    manage_history_array("update")

    if len(history_x_position_array) >= HISTORY_LENGTH*HISTORY_LEVEL and len(history_y_position_array) >= HISTORY_LENGTH*HISTORY_LEVEL and len(history_action_array) >= HISTORY_LENGTH*HISTORY_LEVEL and len(history_image_array) >= HISTORY_LENGTH*HISTORY_LEVEL and len(image_array) >= HISTORY_LENGTH*HISTORY_LEVEL and len(action_array) >= HISTORY_LENGTH*HISTORY_LEVEL:
        print(" "*30 + f"training happend")

        #for index, _ in enumerate(action_array):
        #    if index <= HISTORY_LENGTH:
        #        history_x_position = history_x_position_array[index]
        #        history_y_position = history_y_position_array[index]
        #        history_action = history_action_array[index]
        #        history_image = history_image_array[index]
        #        image = image_array[index]
        #        action = action_array[index]

        #        train_once(history_x_position, history_y_position, history_action, history_image, image, action)

        #manage_history_array("refresh")

        history_x_position = history_x_position_array[0]
        history_y_position = history_y_position_array[0]
        history_action = history_action_array[0]
        history_image = history_image_array[0]
        image = image_array[0]
        action = action_array[0]

        train_once(history_x_position, history_y_position, history_action, history_image, image, action)


while 1:
    #####################
    # FIND A WAY
    ####################
    # take action based on prediction
    if isinstance(image, (np.ndarray, np.generic)):
        if len(history_x_position) >= HISTORY_LENGTH and len(history_y_position) >= HISTORY_LENGTH and len(history_action) >= HISTORY_LENGTH and len(history_image) >= HISTORY_LENGTH:
            action = predict_once(history_x_position, history_y_position, history_action, history_image, image)
        else:
            # take action randomly
            action = env.action_space.sample()
    else:
        # take action randomly
        action = env.action_space.sample()


    if len(history_x_position) >= HISTORY_LENGTH and len(history_y_position) >= HISTORY_LENGTH and len(history_action) >= HISTORY_LENGTH and len(history_image) >= HISTORY_LENGTH:
        # preventing agent learn anything from the initial random actions
        # if initial_flag == True:
        #    manage_history_array("reset")
        #    initial_flag = False

        image_array.append(image.copy())
        history_action_array.append(history_action.copy())
        history_x_position_array.append(history_x_position.copy())
        history_y_position_array.append(history_y_position.copy())
        history_image_array.append(history_image.copy())
        action_array.append(action)

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

        manage_history_array("reset")

        # preventing agent learn anything from the initial random actions
        #initial_flag = True

    #####################
    # THINK ABOUT IT
    ####################
    #if reward >= 0:
    #    # We should think about a strategy to make sure we train it with right data
    #    # Like split history to levels: if we live, for an experience, the older, the better
    #    if isinstance(last_image, (np.ndarray, np.generic)):
    #        if len(history_x_position) >= HISTORY_LENGTH and len(history_y_position) >= HISTORY_LENGTH and len(history_action) >= HISTORY_LENGTH and len(history_image) >= HISTORY_LENGTH:
    #            #train_once(history_x_position, history_y_position, history_action, history_image, image, action)
    #            train_array(history_x_position_array, history_y_position_array, history_action_array, history_image_array, image_array, action_array)

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
