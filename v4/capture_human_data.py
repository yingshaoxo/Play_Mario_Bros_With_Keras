import pickle
import keyboard
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = JoypadSpace(env, SIMPLE_MOVEMENT)


print("""
SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
]
""")

RUN = True

state_array = []
action_array = []

state = None
reward = 0
done = True
info = None

identity = np.identity(env.action_space.n)  # for quickly get a hot vector, like 0001000000000000


def step(action):
    global state, reward, done, info
    global state_array, action_array

    if done:
        state = env.reset()

    if isinstance(state, (np.ndarray, np.generic)):
        state_array.append(state)

    state, reward, done, info = env.step(action)

    action_array.append(identity[action: action+1])

    env.render()


def print_array():
    global RUN

    print('\n'*3)
    print(len(state_array))
    print(len(action_array))

    file_x = open('x.obj', 'wb') 
    pickle.dump(state_array, file_x)

    file_y = open('y.obj', 'wb') 
    pickle.dump(action_array, file_y)

    print("Saved!")
    RUN = False

for num in range(0, 7):
    keyboard.add_hotkey(str(num), step, args=(num, ))
keyboard.add_hotkey("ctrl+shift+esc", print_array)

while RUN:
    pass

env.close()
