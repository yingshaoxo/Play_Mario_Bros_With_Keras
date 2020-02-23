from config import HISTORY_LENGTH  # it's a local py file: config.py
from config import MY_MOVEMENT  # it's a local py file: config.py

import pickle
from model import generate_complex_model
import numpy as np
import os
import tensorflow as tf


epochs = 2


filehandler = open("user_data.obj", 'rb')
user_data = pickle.load(filehandler)


model_file_path = './nn_model.HDF5'
final_model_file_path = './final_nn_model.HDF5'
if os.path.exists(model_file_path):
    model = tf.keras.models.load_model(model_file_path)
else:
    model = generate_complex_model()


history_action_array = user_data['history_action']
history_x_position_array = user_data['history_x_position']
history_y_position_array = user_data['history_y_position']
history_image_array = user_data['history_image']
image_array = user_data['image']
action_array = user_data['action']


number_of_actions = len(MY_MOVEMENT)
identity = np.identity(number_of_actions)  # for quickly get a hot vector, like 0001000000000000


def one_element_of_history_array_to_vector(one_element_of_history_array, only_value=False):
    one_element_of_history_array = one_element_of_history_array[:HISTORY_LENGTH]
    vector = []
    for item in one_element_of_history_array:
        value = item["value"]
        times = item["times"]
        vector.append(value)
        if only_value == False:
            vector.append(times)
    return vector


array_history_x_position = []
array_history_y_position = []
array_history_action = []
array_history_image = []
array_image = []
array_action = []
for index, _ in enumerate(image_array):
    history_x_position = one_element_of_history_array_to_vector(history_x_position_array[index])
    history_y_position = one_element_of_history_array_to_vector(history_y_position_array[index])
    history_action = one_element_of_history_array_to_vector(history_action_array[index])
    history_image = one_element_of_history_array_to_vector(history_image_array[index], only_value=True)
    image = image_array[index]

    try:
        action = action_array[index]
        action = identity[action:action+1][0]
    except Exception as e:
        print(e)
        # we think the action we collected before should be in the new nerual model, so we ignore(skip) it
        continue

    array_history_x_position.append(history_x_position)
    array_history_y_position.append(history_y_position)
    array_history_action.append(history_action)
    array_history_image.append(history_image)
    array_image.append(image)
    array_action.append(action)


for current_epoch in range(epochs+1):
    model.fit(
        x={
            'history_x_position': np.array(array_history_x_position),
            'history_y_position': np.array(array_history_y_position),
            'history_action': np.array(array_history_action),
            'history_image': np.array(array_history_image),
            'image': np.array(array_image),
        },
        y=np.array(array_action),
    )

    print("saving model...")
    model_file_path = './nn_model.HDF5'
    model.save(model_file_path)
