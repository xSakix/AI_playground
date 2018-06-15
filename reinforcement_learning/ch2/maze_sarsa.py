import pandas as pd
import numpy as np
import time
import sys
import tkinter as tk

UNIT = 40  # pixels
MAZE_H = 4
MAZE_W = 4


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.actions = ['u', 'd', 'r', 'l']
        self.n_actions = len(self.actions)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self._build_maze()

    def _create_rectangle(self, center, color='black'):
        return self.canvas.create_rectangle(
            center[0] - 15,
            center[1] - 15,
            center[0] + 15,
            center[1] + 15,
            fill=color)

    def _create_oval(self, center, color='yellow'):
        return self.canvas.create_oval(
            center[0] - 15,
            center[1] - 15,
            center[0] + 15,
            center[1] + 15,
            fill=color)

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)
        # grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([20, 20])

        # hell (?:))
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self._create_rectangle(hell1_center)
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self._create_rectangle(hell2_center)

        # oval
        oval_center = origin + UNIT * 2
        self.oval = self._create_oval(oval_center)

        # red rectangle
        self.rect = self._create_rectangle(origin, color='red')

        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self._create_rectangle(origin, color='red')

        return self.canvas.coords(self.rect)

    def step(self, action):
        state = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if state[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if state[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT

        # move agent
        self.canvas.move(self.rect, base_action[0], base_action[1])

        new_state = self.canvas.coords(self.rect)
        if new_state == self.canvas.coords(self.oval):
            reward = 1
            done = True
            new_state = 'terminal'
        elif new_state in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            new_state = 'terminal'
        else:
            reward = 0
            done = False

        return new_state, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


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


class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, state, action, reward, new_state, new_action):
        self.check_state_exists(new_state)
        q_predict = self.q_table.loc[state, action]

        if new_state != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[new_state, new_action]
        else:
            q_target = reward

        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

    def __str__(self) -> str:
        return str(self.q_table)


def update():
    for episode in range(100):
        state = env.reset()
        action = RL.choose_action(str(state))

        print('\r [%d] %s -> %s ' % (episode,state, env.actions[action]),end='')

        while True:
            env.render()
            new_state, reward, done = env.step(action)
            new_action = RL.choose_action(str(new_state))
            RL.learn(
                str(state),
                action,
                reward,
                str(new_state),
                new_action

            )

            state = new_state
            action = new_action
            print('\r [%d] %s -> %s ' % (episode, state, env.actions[action]), end='')

            if done:
                break

    print('game over')
    print(RL.q_table)
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
