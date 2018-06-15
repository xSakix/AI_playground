import gym

env = gym.make('SpaceInvaders-v0')

for _ in range(1000):
    env.reset()
    env.render()
    done = False
    while not done:
        observation, reward, done, info = env.step(env.action_space.sample())
        env.render()

env.close()
