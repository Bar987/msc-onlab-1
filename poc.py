from puzzle_gym.envs.puzzle_env import PuzzleEnv

IMAGE_DIR = '../testSample'

env = PuzzleEnv(img_dir=IMAGE_DIR,
                img_size=(36, 36), puzzle_size=(3, 3), puzzle_type="slide", dist_type="manhattan", penalty_for_step=-0.15,
                reward_for_completiton=20, positive_reward_coefficient=1.0)
env.render()
done = False
while not done:
    n = int(input("Enter n: "))
    _, rew, done, ___ = env.step(n)
    print(rew)
