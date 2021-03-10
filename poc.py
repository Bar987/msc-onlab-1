from puzzle_gym.envs.puzzle_env import PuzzleEnv

env = PuzzleEnv(img_size=(9, 9), puzzle_size=(3, 3))
env.render()
while True:
    n = int(input("Enter n: "))
    env.step(n)
    env.render()
