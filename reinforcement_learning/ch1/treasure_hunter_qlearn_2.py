import warnings

warnings.filterwarnings('ignore')

import time
import pandas as pd
import numpy as np

EPSILON = 0.6  # greedy
ALPHA = 0.1
GAMMA = 0.3
ACTIONS = ['left', 'right']
N_STATES = 8
FRESH_TIME = 0.3
MAX_EPISODES = 13


class RL:
    def __init__(self, action_space, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exists(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        # print('\n choosing action: %s' % observation)
        self.check_state_exists(observation)
        if np.random.rand() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            # print('\nstate actions for %s is %s' % (observation, state_action))
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            # print('\nstate actions after reindex for %s is %s' % (observation, state_action))
            action = state_action.idxmax()
            # print('\nnew action is %s' % action)

        else:
            action = np.random.choice(self.actions)
            # print('\n choosing random action:%s' % action)

        return action

    def learn(self, *args):
        pass


class QLearningTable(RL):

    def __init__(self, action_space, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(action_space, learning_rate, reward_decay, e_greedy)

    def learn(self, state, action, reward, new_state):
        self.check_state_exists(new_state)
        q_predict = self.q_table.loc[state, action]

        if new_state != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[new_state, :].max()
        else:
            q_target = reward

        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

    def __str__(self) -> str:
        return str(self.q_table)


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


def update_env(state, episode, step_counter, action='right'):
    env_list = ['-'] * (N_STATES - 1) + ['T']  # -----T is our environment
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                               ', end='')
    else:
        character = 'o'
        if action == 'up':
            character = 'u'
        if action == 'down':
            character = 'd'
        env_list[state] = character
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = QLearningTable(ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminated = False
        update_env(state, episode, step_counter)

        while not is_terminated:
            action = q_table.choose_action(state)
            new_state, reward = get_env_feedback(state, action)
            q_table.learn(state, action, reward, new_state)

            if new_state == 'terminal':
                is_terminated = True

            state = new_state

            step_counter += 1
            update_env(state, episode, step_counter, action)

    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-TABLE:\n')
    print(q_table)
