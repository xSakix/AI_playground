from reinforcement_learning.crypto_market.util import State
import sys

from reinforcement_learning.reinforce import policy, policy_evaluator, cont_policy_evaluator

sys.path.insert(0, '../../../etf_data')
from etf_data_loader import load_all_data_from_file2
import numpy as np
import matplotlib.pyplot as plt


def run_agent():
    start_date = '2018-04-01'
    end_date = '2018-09-14'
    # start_date = '2011-01-01'
    # end_date = '2018-04-01'


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

    # model = policy.create_dense_model(0.001)
    timesteps = 7
    model = policy.create_lstm_model(0.001,timesteps)
    model.load_weights('weights.h5')
    # x = states.get_whole_state()[window:]
    # x = np.reshape(x, (x.shape[0], timesteps, x.shape[1]))
    input = []
    for t in range(window + timesteps, len(data)):
        x = states.get_partial_state(t, timesteps)
        x = x.reshape((1, timesteps, 5))
        input.append(x)

    x = np.array(input).reshape((len(input), timesteps, 5))

    predicted_action_proba = model.predict(x)
    print(predicted_action_proba.shape)
    print(x.shape)
    # actions = np.empty(predicted_action_proba.shape[0], dtype=np.int32)
    agent_evaluator = cont_policy_evaluator.RecordingAgent(data, states)
    # for i in range(predicted_action_proba.shape[0]):
    #     actions[i] = np.random.choice(3, 1, p=predicted_action_proba[i])
    for run in range(predicted_action_proba.shape[0]):
        action = np.random.choice(3, 1, p=predicted_action_proba[run])[0]
        agent_evaluator.run(action, window+run)

    # agent_evaluator = policy_evaluator.Agent(actions)
    # agent_evaluator.run(data, states)

    print(agent_evaluator.ror_history[-1])

    plt.plot(agent_evaluator.ror_history)
    plt.title('ror')
    plt.show()

    _,ax = plt.subplots(2,1)
    ax[0].plot(agent_evaluator.rewards)
    ax[1].plot(agent_evaluator.disco_rewards)
    plt.show()


if __name__ == '__main__':
    run_agent()
