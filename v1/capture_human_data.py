from config import HISTORY_LENGTH

import pickle
import keyboard
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from pprint import pprint
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

Press NumPad key from 0 to 7 to control your game.
You probobaly need to press 1 for a long time to get started!
""")

RUN = True

history_action_array = []
history_x_position_array = []
history_y_position_array = []
history_image_array = []
image_array = []
action_array = []

state = None
reward = 0
done = True
info = None

identity = np.identity(env.action_space.n)  # for quickly get a hot vector, like 0001000000000000

history_action = []
history_x_position = []
history_y_position = []
history_image = []


def step(action):
    global HISTORY_LENGTH
    global state, reward, done, info
    global action_array
    global history_action_array, history_x_position_array, history_y_position_array, image_array, history_image_array
    global history_action, history_x_position, history_y_position, history_image

    if done:
        state = env.reset()

    if isinstance(state, (np.ndarray, np.generic)):
        if len(history_action) == HISTORY_LENGTH and len(history_x_position) == HISTORY_LENGTH and len(history_y_position) == HISTORY_LENGTH and len(history_image) == HISTORY_LENGTH:
            image_array.append(state.copy())

            history_action_array.append(history_action.copy())
            history_x_position_array.append(history_x_position.copy())
            history_y_position_array.append(history_y_position.copy())
            history_image_array.append(history_image.copy())

            action_array.append(action)

    state, reward, done, info = env.step(action)
    env.render()

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
        history_image.append({"value": state.copy(), "times": 0})
    else:
        if np.array_equal(state.copy(), history_image[-1]['value']):
            history_image[-1]["times"] += 1
        else:
            history_image.append({"value": state.copy(), "times": 0})
        history_image = history_image[-HISTORY_LENGTH:]


def print_array():
    global RUN

    print('\n'*3)
    print(len(image_array))

    print('\n'*3)
    pprint(history_action)

    file_x = open('user_data.obj', 'wb')
    pickle.dump({
        "history_action": history_action_array,
        "history_x_position": history_x_position_array,
        "history_y_position": history_y_position_array,
        "image": image_array,
        "history_image": history_image_array,
        "action": action_array
    }, file_x)

    print("Saved!")
    RUN = False


for num in range(0, 7):
    keyboard.add_hotkey(str(num), step, args=(num, ))
keyboard.add_hotkey("ctrl+shift+esc", print_array)

while RUN:
    pass

env.close()
