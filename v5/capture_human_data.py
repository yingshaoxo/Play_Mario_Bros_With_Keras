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
history_actions_array = []
history_x_pos_array = []
history_y_pos_array = []

state = None
reward = 0
done = True
info = None

identity = np.identity(env.action_space.n)  # for quickly get a hot vector, like 0001000000000000

history_actions = []
history_x_pos = []
history_y_pos = []


def step(action):
    global state, reward, done, info
    global state_array, action_array, history_actions_array, history_x_pos_array, history_y_pos_array
    global history_actions, history_x_pos, history_y_pos

    if done:
        state = env.reset()

    if isinstance(state, (np.ndarray, np.generic)):
        if len(history_actions) == 32 and len(history_x_pos) == 32 and len(history_y_pos) == 32:
            state_array.append(state)
            history_actions_array.append(history_actions.copy())
            history_x_pos_array.append(history_x_pos.copy())
            history_y_pos_array.append(history_y_pos.copy())

    state, reward, done, info = env.step(action)

    history_actions.append(action)
    history_actions = history_actions[-32:]
    history_x_pos.append(info['x_pos'])
    history_x_pos = history_x_pos[-32:]
    history_y_pos.append(info['y_pos'])
    history_y_pos = history_y_pos[-32:]

    if len(history_actions) == 32 and len(history_x_pos) == 32 and len(history_y_pos) == 32:
	action_array.append(identity[action: action+1])

    env.render()


def print_array():
    global RUN

    print('\n'*3)
    print(len(state_array))
    print(len(action_array))

    file_x = open('user_data.obj', 'wb')
    pickle.dump({
        "img": state_array,
        "action": action_array,
        "history_actions": history_actions_array,
        "history_x_pos": history_x_pos_array,
        "history_y_pos": history_y_pos_array,
    }, file_x)

    print("Saved!")
    RUN = False


for num in range(0, 7):
    keyboard.add_hotkey(str(num), step, args=(num, ))
keyboard.add_hotkey("ctrl+shift+esc", print_array)

while RUN:
    pass

env.close()
