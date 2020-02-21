from model import generate_complex_model
import numpy as np


import pickle
filehandler = open("user_data.obj", 'rb')
user_data = pickle.load(filehandler)


model = generate_complex_model()


history_action_array = user_data['history_action']
history_x_position_array = user_data['history_x_position']
history_y_position_array = user_data['history_y_position']
image_array = user_data['image']
#history_image_array = user_data['history_image']
action_array = user_data['action']


number_of_actions = 7
identity = np.identity(number_of_actions)  # for quickly get a hot vector, like 0001000000000000


def one_element_of_history_array_to_vector(one_element_of_history_array):
    vector = []
    for item in one_element_of_history_array:
        value = item["value"]
        times = item["times"]
        vector.append(value)
        vector.append(times)
    return vector


for index, _ in enumerate(image_array):
    history_x_position = one_element_of_history_array_to_vector(history_x_position_array[index])
    history_y_position = one_element_of_history_array_to_vector(history_y_position_array[index])
    history_action = one_element_of_history_array_to_vector(history_action_array[index])
    #history_image = one_element_of_history_array_to_vector(history_image_array[index])
    image = image_array[index]

    action = action_array[index]
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


model_file_path = './nn_model.HDF5'
model.save(model_file_path)
