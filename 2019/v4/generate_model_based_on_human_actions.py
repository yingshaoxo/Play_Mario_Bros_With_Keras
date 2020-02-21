from model import generate_model
import tensorflow as tf
import numpy as np
import os


import pickle
filehandler = open("x.obj", 'rb')
state = pickle.load(filehandler)

filehandler = open("y.obj", 'rb')
action = pickle.load(filehandler)


model_file_path = './nn_model.HDF5'
if os.path.exists(model_file_path):
    model = tf.keras.models.load_model(model_file_path)
else:
    model = generate_model()


for x, y in zip(state, action):
    model.train_on_batch(x=np.expand_dims(x, axis=0), y=y)


model.save(model_file_path)
