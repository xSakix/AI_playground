from reinforcement_learning.crypto_market.util import State
import sys

from reinforcement_learning.reinforce import policy, policy_evaluator

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2
import numpy as np
import matplotlib.pyplot as plt


def run_agent():
    start_date = '2011-01-01'
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

    print(data)

    window = 30
    states = State(window, data)
    print(states.bench[0])
    print(states.bench[1])
    print(states.bench[-2])
    print(states.bench[-1])
    print(len(data) - window)
    learning_rate = 0.1
    # model = policy.create_lstm_model(learning_rate)
    model = policy.create_dense_model(learning_rate)
    x = states.get_whole_state()[window:]
    # x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    rors = []
    losses = []
    discos = []
    print()
    max_ror = None

    for it in range(1000):
        predicted_action_proba = model.predict(x)
        actions = np.empty(predicted_action_proba.shape[0], dtype=np.int32)

        for i in range(predicted_action_proba.shape[0]):
            actions[i] = np.random.choice(3, 1, p=predicted_action_proba[i])

        agent_evaluator = policy_evaluator.Agent(actions)
        agent_evaluator.run(data, states)

        rors.append(agent_evaluator.ror_history[-1])

        if max_ror is None or max_ror < rors[it]:
            max_ror = rors[it]
            print('saving at ror:',rors[it])
            model.save_weights('weights_temp.h5', overwrite=True)

        disco = agent_evaluator.disco_rewards - np.mean(agent_evaluator.disco_rewards)
        disco = disco / np.std(agent_evaluator.disco_rewards)
        # disco = np.reshape(disco, (disco.shape[0], 1))
        discos.append(disco[-1])
        # y = predicted_action_proba + learning_rate * disco

        y = np.zeros_like(predicted_action_proba)
        for i in range(predicted_action_proba.shape[0]):
            y[i][actions[i]] = disco[i]

        loss = model.fit(x, y,
                         nb_epoch=10,
                         verbose=0,
                         shuffle=True,
                         validation_split=0.3)
        losses.append(loss.history['loss'])
        print('\r[%d] %f | %f | %f | %f' % (it, rors[it], losses[it][-1], agent_evaluator.rewards[-1], disco[-1]),
              end='')

        # if loss.history['loss'][-1] <= 0. or np.isnan(loss.history['loss'][-1]):
        #     break
        if np.isnan(loss.history['loss'][-1]):
            print('loading model...')
            model.load_weights('weights_temp.h5')

    model.save_weights('weights.h5', overwrite=True)

    _, ax = plt.subplots(3, 1)
    ax[0].plot(rors)
    ax[0].set_title('rors')
    ax[1].plot(losses)
    ax[1].set_title('loss')
    ax[2].plot(discos)
    ax[2].set_title('disco')

    plt.show()


if __name__ == '__main__':
    run_agent()
