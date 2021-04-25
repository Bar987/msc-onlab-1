from puzzle_gym.envs.puzzle_env import PuzzleEnv

env = PuzzleEnv(images=None,
                img_size=(36, 36),channel_num=1, puzzle_size=(3, 3), puzzle_type="switch", dist_type="manhattan", penalty_for_step=-0.15,
                reward_for_completiton=20, positive_reward_coefficient=1.0, render_mode="human")

env.render()
done = False
while not done:
    n = int(input("Enter n: "))
    _, rew, done, ___ = env.step(n)
    print(rew)
