import warnings

warnings.filterwarnings('ignore')

import time
import pandas as pd
import numpy as np

EPSILON = 0.6 # greedy
ALPHA = 0.1
GAMMA = 0.3
ACTIONS = ['left', 'right']
N_STATES = 8
FRESH_TIME = 0.3
MAX_EPISODES = 13


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions)
    print(table)
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]

    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()

    return action_name


def get_env_feedback(state, action):
    if action == 'right':
        if state == N_STATES - 2:  # terminate
            new_state = 'terminal'
            reward = 1
        else:
            new_state = state + 1
            reward = 0
    else:
        reward = 0
        if state == 0:  # reached the wall
            new_state = state
        else:
            new_state = state - 1  # guess he moved left?

    return new_state, reward


def update_env(state, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']  # -----T is our environment
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                               ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminated = False
        update_env(state, episode, step_counter)

        while not is_terminated:
            action = choose_action(state, q_table)
            new_state, reward = get_env_feedback(state, action)
            q_predict = q_table.ix[state, action]
            if new_state != 'terminal':
                q_target = reward + GAMMA * q_table.iloc[new_state, :].max()
            else:
                q_target = reward
                is_terminated = True

            q_table.ix[state, action] += ALPHA * (q_target - q_predict)
            state = new_state

            step_counter += 1
            update_env(state, episode, step_counter)

    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-TABLE:\n')
    print(q_table)
