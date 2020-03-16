from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.agents.dqn import DQNAgent

import numpy as np
#import tensorflow as tf
import keras

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace


env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env,  [
    ['right', 'A'],
    ['right'],
])
env.reset()
np.random.seed(123)
env.seed(123)

# base_model = keras.applications.MobileNetV2(
#    input_shape=env.observation_space.shape,
#    include_top=False,
#    weights='imagenet'
# )
#base_model.trainable = False
#image_x = keras.layers.Flatten()(base_model.outputs[0])
#image_x = keras.layers.Dense(1024, activation='sigmoid')(image_x)
#image_x = keras.layers.Dropout(0.5)(image_x)
#image_x = keras.layers.Dense(512, activation='sigmoid')(image_x)
#image_x = keras.layers.Dropout(0.5)(image_x)
#image_x = keras.layers.Dense(128, activation='sigmoid')(image_x)
#image_x = keras.layers.Dropout(0.5)(image_x)
#image_outputs = keras.layers.Dense(env.action_space.n, activation='linear', name="image_outputs")(image_x)
# model = keras.Model(
#    inputs=base_model.input,
#    outputs=image_outputs
# )

image_inputs = keras.layers.Input((1,) + env.observation_space.shape)
image_x = keras.layers.ConvLSTM2D(32, (3, 2), activation='relu')(image_inputs)
image_x = keras.layers.Conv2D(64, (3, 2), activation='relu')(image_x)
image_x = keras.layers.Conv2D(64, (3, 2), activation='relu')(image_x)
image_x = keras.layers.Flatten()(image_x)
image_x = keras.layers.Dense(512, activation='relu')(image_x)
image_outputs = keras.layers.Dense(env.action_space.n, activation='linear', name="image_outputs")(image_x)
model = keras.Model(
    inputs=image_inputs,
    outputs=image_outputs
)


memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=500, target_model_update=1e-2, policy=policy)
dqn.compile(keras.optimizers.Adam(lr=1e-3), metrics=['mae'])


dqn.fit(env, nb_steps=50000, visualize=True, verbose=1)

dqn.save_weights('dqn_mario_weights.h5f', overwrite=True)

dqn.test(env, nb_episodes=1, visualize=True)
