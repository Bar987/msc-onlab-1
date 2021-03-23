import gym
from puzzle_gym.envs.puzzle_env import PuzzleEnv
import puzzle_gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

IMAGE_DIR = '../../testSample'

env = Monitor(gym.make('puzzle_gym:puzzle-v0', img_dir=IMAGE_DIR,
                       img_size=(36, 36), puzzle_size=(3, 3), puzzle_type="switch", dist_type="manhattan", penalty_for_step=-0.2,
                       reward_for_completiton=20, positive_reward_coefficient=1.0))

model = DQN('CnnPolicy', env, verbose=1, learning_rate=0.001, seed=42)
model.learn(total_timesteps=150000)

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
print(rews, i, sum(rews))
