import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam

import matplotlib.pyplot as plt

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000


def some_random_games_first():
    for _ in range(5):
        for t in range(200):
            env.reset()
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break


# some_random_games_first()


def generate_traning_data():
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []

        for step in range(goal_steps):
            # print('\r %d/%d ' % (step, goal_steps), end='')
            # action = random.randrange(0, 2)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break
        # print('\r %s' % str(game_memory), end='')
        if score > score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                training_data.append([data[0], output])
        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Avg score: ', mean(accepted_scores))
    print('Median score: ', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def neural_network_model(input_size):
    print(str([None, input_size, 1]))

    model = Sequential()
    model.add(Dense(input_size * 2 + 1, activation='relu', input_shape=(input_size, 1)))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(lr=LR), loss='categorical_crossentropy', metrics=['acc'])

    print(model.summary())

    return model


def train_model(training_data, model=False):
    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = np.array([i[1] for i in training_data])
    print('x shape = ', x.shape)
    print('y shape = ', y.shape)

    print(training_data[0])

    if not model:
        model = neural_network_model(input_size=len(x[0]))

    history_o = model.fit(
        x,
        y,
        epochs=20,
        validation_split=0.3,
        shuffle=True,
        batch_size=128
    )

    plt.plot(history_o.history['loss'])
    plt.plot(history_o.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()

    return model


training_data = generate_traning_data()
# training_data = np.load('saved.npy')
model = train_model(training_data)
scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
        scores.append(score)

print('Avg score:', sum(scores) / len(scores))
print('choice 1:{} choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
print(score_requirement)

env.close()
