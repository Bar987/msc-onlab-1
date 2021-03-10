import gym
from puzzle_gym.envs.puzzle_env import PuzzleEnv
import puzzle_gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make('puzzle_gym:puzzle-v0',
                       img_size=(12, 12), puzzle_size=(3, 3), penalty_for_step=-0.15,
                       reward_for_completiton=1, positive_reward_coefficient=1.0))

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
done = False
rews = []
i = 0
while not done:
    i += 1
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    rews.append(rewards)
    print(rewards)
    env.render()
print(rews, i)
