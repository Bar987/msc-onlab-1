from gym.envs.registration import register

register(
    id='puzzle-v0',
    entry_point='puzzle_gym.envs:PuzzleEnv',
)
