import pandas as pd
from keras.utils import to_categorical

from reinforcement_learning.crypto_market.util import State
import sys

from reinforcement_learning.reinforce import policy, policy_evaluator, cont_policy_evaluator

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2
import numpy as np
import matplotlib.pyplot as plt

import warnings
from scipy import special

warnings.filterwarnings("ignore")


def run_agent():
    # start_date = '2011-01-01'
    start_date = '2018-01-01'
    end_date = '2018-09-14'

    ticket = 'BTC-EUR'

    data = load_all_data_from_file2('btc_etf_data_adj_close.csv', start_date, end_date)
    data = data[data['date'] > str(start_date)]
    data = data[data['date'] < str(end_date)]

    print(start_date, " - ", end_date, " ,len = ", len(data))
    data = data[data[ticket] > 0.]
    data = data.reindex(method='bfill')
    data.reset_index(inplace=True)
    data = data[ticket]

    window = 30
    learning_rate = 0.001

    timesteps = 7

    model = policy.create_lstm_model(learning_rate, timesteps)
    # model = policy.create_dense_model(learning_rate)
    # x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    all_rors = []
    all_losses = []
    all_discos = []
    print()
    print('[episode][it/max it] ror | loss | reward | expected_reward | action')
    actions = {0: 'hold', 1: 'sell', 2: 'buy'}
    states = State(window, data)
    for episode in range(10):
        input = []
        labels = []
        losses = []
        discos = []
        rors = []

        for t in range(window + timesteps, len(data)):
            agent_evaluator = cont_policy_evaluator.RecordingAgent(data, states)
            # x = states.get_state(t)
            x = states.get_partial_state(t, timesteps)
            # lstm
            x = x.reshape((1, timesteps, 5))
            input.append(x)
            x = np.array(input).reshape((len(input), timesteps, 5))
            # dense input
            # x = x.reshape((1, 5))
            predicted_action_proba = model.predict(x)
            runs = predicted_action_proba.shape[0]-1
            for run in range(predicted_action_proba.shape[0]):
                action = np.random.choice(3, 1, p=predicted_action_proba[run])[0]
                agent_evaluator.run(action, t-runs+run)
                # print(run, '|', action, '|', agent_evaluator.rewards[t - window-runs+run])

            index = t - window

            rors.append(agent_evaluator.ror_history[index])
            discos.append(agent_evaluator.disco_rewards[-1])

            # y = predicted_action_proba + learning_rate * agent_evaluator.disco_rewards
            y = predicted_action_proba * agent_evaluator.disco_rewards
            # print(y.shape)
            # labels.append(y.reshape((3,)))
            # y = np.array(labels)

            loss = model.fit(x, y,
                             nb_epoch=1,
                             verbose=0,
                             shuffle=True,
                             validation_split=0.3)

            if 'loss' in loss.history.keys():
                losses.append(loss.history['loss'])
                print('\r[%d][%d/%d] %f | %f | %f | %f | %s' % (
                    episode, t, len(data), rors[-1], losses[-1][-1], np.mean(agent_evaluator.rewards),
                    agent_evaluator.disco_rewards[-1],
                    actions[action]),
                      end='')
        all_losses.append(losses)
        all_discos.append(discos)
        all_rors.append(rors)

    model.save_weights('weights.h5', overwrite=True)

    _, ax = plt.subplots(3, 1)
    for ii in range(len(all_rors)):
        ax[0].plot(all_rors[ii], label=str(ii))
    ax[0].set_title('rors')
    for ii in range(len(all_losses)):
        ax[1].plot(all_losses[ii], label=str(ii))
    ax[1].set_title('loss')
    for ii in range(len(all_discos)):
        ax[2].plot(all_discos[ii], label=str(ii))
    ax[2].set_title('expected_reward')
    for axis in ax:
        axis.legend()

    plt.show()


if __name__ == '__main__':
    run_agent()
