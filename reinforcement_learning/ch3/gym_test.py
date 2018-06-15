import gym

env = gym.make('CartPole-v0')
env.reset()

for _ in range(10):
    env.reset()
    env.render()
    done = False
    while not done:
        observation, reward, done, info = env.step(env.action_space.sample())
        env.render()

env.close()
