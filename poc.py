from puzzle_gym.envs.puzzle_env import PuzzleEnv

env = PuzzleEnv(img_size=(9, 9), puzzle_size=(3, 3))
env.render()
done = False
while not done:
    n = int(input("Enter n: "))
    _, rew, done, ___ = env.step(n)
    print(rew)
