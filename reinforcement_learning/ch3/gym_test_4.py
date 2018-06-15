import gym
import gym.benchmarks
import universe

env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)

# for _ in range(10):
#     env.reset()
#     env.render()
#     done = False
#     while not done:
#         observation, reward, done, info = env.step(env.action_space.sample())
#         env.render()

env.close()
