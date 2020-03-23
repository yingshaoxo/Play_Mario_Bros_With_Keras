import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import gym

from config import HISTORY_LENGTH  # it's a local py file: config.py
from config import MY_MOVEMENT  # it's a local py file: config.py

import numpy as np
import random
import os
from collections import deque

from tensorflow import keras
import redis
r = redis.Redis(host='localhost', port=6379, db=0)


model_file_path = './nn_model'
final_model_file_path = './final_nn_model'

test = True


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85

        epsilon = float(r.get("epsilon").decode("utf-8"))
        if epsilon == None:
            r.set("epsilon", 1.0)
            epsilon = float(r.get("epsilon").decode("utf-8"))
        self.epsilon = epsilon

        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        if os.path.exists(model_file_path):
            model = keras.models.load_model(model_file_path)
        else:
            base_model = keras.applications.MobileNetV2(input_shape=self.env.observation_space.shape,
                                                        include_top=False,
                                                        weights='imagenet'
                                                        )
            base_model.trainable = False
            x = keras.layers.Flatten()(base_model.outputs[0])
            x = keras.layers.Dense(32, activation='relu')(x)
            x = keras.layers.Dense(32, activation='relu')(x)
            x = keras.layers.Dense(32, activation='relu')(x)
            outputs = keras.layers.Dense(self.env.action_space.n)(x)

            model = keras.Model(
                inputs=base_model.input,
                outputs=outputs
            )

            model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate),
                          loss="mean_squared_error",
                          metrics=['accuracy'])
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if not test:
            r.set("epsilon", str(self.epsilon))
        if test:
            self.epsilon = 0.2
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
            print(f"random: {action}")
        else:
            action = np.argmax(self.model.predict(np.expand_dims(state, axis=0))[0])
            print(f"predict: {action}")
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = HISTORY_LENGTH
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(np.expand_dims(state, axis=0))
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(np.expand_dims(new_state, axis=0))[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(np.expand_dims(state, axis=0), target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env,  MY_MOVEMENT)
    env.reset()
    env = gym.wrappers.Monitor(env, directory='./video', force=True)

    gamma = 0.9
    epsilon = .95

    trials = 5000
    trial_len = 500

    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        last_state = env.reset()
        print("It's in {} trials".format(trial))
        for step in range(trial_len):
            action = dqn_agent.act(last_state)

            if action == 0:
                new_state, reward, done, info = env.step(1)
            else:
                new_state, reward, done, info = env.step(0)
            if not done:
                for _ in range(20):
                    new_state, reward, done, info = env.step(action)
                    env.render()
                    if done:
                        break

            if not test:
                dqn_agent.remember(last_state, action, reward, new_state, done)

                dqn_agent.replay()       # internally iterates default (prediction) model
                dqn_agent.target_train()  # iterates target model

            last_state = new_state

            if done:
                break

        if not test:
            print("model saving...")
            dqn_agent.save_model(model_file_path)


if __name__ == "__main__":
    main()
