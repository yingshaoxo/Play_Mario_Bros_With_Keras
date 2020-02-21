from model import generate_model
import tensorflow as tf
import numpy as np
import os


import pickle
filehandler = open("user_data.obj", 'rb')
user_data = pickle.load(filehandler)

model_file_path = './nn_model.HDF5'
if os.path.exists(model_file_path):
    model = tf.keras.models.load_model(model_file_path)
else:
    model = generate_model()


img = user_data['img']
action = user_data['action']
history_actions = user_data['history_actions']
history_x_pos = user_data['history_x_pos']
history_y_pos = user_data['history_y_pos']

for index, _ in enumerate(img):
    model.train_on_batch(
        x={
            'img': np.expand_dims(img[index], axis=0),
            'action': np.expand_dims(np.array(history_actions[index]), axis=0),
            'x_position': np.expand_dims(np.array(history_x_pos[index]), axis=0),
            'y_position': np.expand_dims(np.array(history_y_pos[index]), axis=0),
        },
        y=action[index]
    )


model.save(model_file_path)
