import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from math import prod, sqrt

CHANNEL_NUM = 1


class PuzzleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, img_path=None, img_size=(100, 100), puzzle_size=(2, 2), penalty_for_step=-0.1, reward_for_completiton=1, positive_reward_coefficient=1.0):

        self.puzzle_size = puzzle_size
        self.img_path = img_path

        self.tile_size = max(
            img_size[0] // puzzle_size[0], img_size[1] // puzzle_size[1])

        self.img_size = (
            self.puzzle_size[0] * self.tile_size, self.puzzle_size[1] * self.tile_size)

        self.action_space = spaces.Discrete(self.puzzle_size[0] * (self.puzzle_size[1] - 1)
                                            + self.puzzle_size[1] * (self.puzzle_size[0] - 1))

        self.observation_space = spaces.Box(low=0, high=255, shape=(
            self.puzzle_size[0], self.puzzle_size[1], self.tile_size, self.tile_size, CHANNEL_NUM), dtype=np.uint8)

        self.penalty_for_step = penalty_for_step
        self.reward_for_completiton = reward_for_completiton
        self.positive_reward_coefficient = positive_reward_coefficient

        self.reset()

    def step(self, action):
        self.last_reward = self.penalty_for_step

        temp = self.puzzle_size[0] * (self.puzzle_size[0]-1)
        if action < temp:
            divident = action // (self.puzzle_size[1]-1)
            stride = action % (self.puzzle_size[1]-1)
            idx = divident * self.puzzle_size[1] + stride
            reward = self._swap_elements(idx, idx + 1)
        else:
            idx = action % temp
            reward = self._swap_elements(idx, idx + self.puzzle_size[1])

        if reward > 0:
            reward = self.positive_reward_coefficient * reward

        self.last_reward += reward

        if self._is_done():
            done = True
            self.last_reward += self.reward_for_completiton
        else:
            done = False

        self.render()

        observation = self._get_observation_from_puzzle()
        info = {}
        return observation, self.last_reward, done, info

    def reset(self):
        self._init_puzzle()
        self.last_reward = 0
        return self._get_observation_from_puzzle()

    def render(self, mode='human'):
        for i in range(self.puzzle_size[0]):
            for j in range(self.puzzle_size[1]):
                idx = i * self.puzzle_size[1] + j
                print('{0:2d}'.format(
                    self.puzzle[idx].correct_idx), end="  ")
            print('\n', end="")
        # for i in range(self.puzzle_size[0]):
        #     for j in range(self.puzzle_size[1]*self.tile_size):
        #         idx = i * self.puzzle_size[1] + j % self.puzzle_size[1]
        #         k = j // self.puzzle_size[1]
        #         for m in range(self.tile_size):
        #             print('{0:2d}'.format(
        #                 self.puzzle[idx].tile[k][m][0]), end="  ")
        #         if j % self.puzzle_size[1] == self.puzzle_size[1] - 1:
        #             print('\n', end="")
        #         else:
        #             print('\t', end="")
        #     print('\n', end="")

    def _init_puzzle(self):
        if not self.img_path:
            puzzle = np.arange(self.tile_size**2 * prod(self.puzzle_size)).repeat(
                CHANNEL_NUM).reshape(self.img_size[0], self.img_size[1], CHANNEL_NUM)

            puzzle = np.array(np.hsplit(puzzle, self.puzzle_size[1]))
            puzzle = np.array(np.hsplit(puzzle, self.puzzle_size[0]))

        self.puzzle = []

        for i, row in enumerate(puzzle):
            for j, tile in enumerate(row):
                idx = i * self.puzzle_size[1] + j
                self.puzzle.append(Tile(tile, idx, idx))

        np.random.shuffle(self.puzzle)

        for i, tile in enumerate(self.puzzle):
            tile.set_current_idx(i)

    def _swap_elements(self, idx1, idx2):
        score_before = self._calculate_reward(idx1, idx2)
        temp = self.puzzle[idx1]
        self.puzzle[idx1] = self.puzzle[idx2]
        self.puzzle[idx2] = temp
        score_after = self._calculate_reward(idx1, idx2)

        return score_before - score_after

    def _is_done(self):
        for i, tile in enumerate(self.puzzle):
            if i != tile.correct_idx:
                return False
        return True

    def _calculate_reward(self, idx1, idx2):
        res = 0
        for i in [idx1, idx2]:
            correct_coord = (
                self.puzzle[i].correct_idx // self.puzzle_size[1], self.puzzle[i].correct_idx % self.puzzle_size[1])
            current_coord = (
                i // self.puzzle_size[1], i % self.puzzle_size[1])
            distance = self._calculate_point_dist(correct_coord, current_coord)
            res += distance

        return res

    def _calculate_point_dist(self, p1, p2):
        return sqrt(
            ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

    def _get_observation_from_puzzle(self):
        observation = np.zeros(shape=(
            self.puzzle_size[0], self.puzzle_size[1], self.tile_size, self.tile_size, CHANNEL_NUM))

        for idx, puzzle in enumerate(self.puzzle):
            j = idx % self.puzzle_size[1]
            i = idx // self.puzzle_size[1]
            observation[i, j] = puzzle.tile

        return observation


class Tile:
    def __init__(self, tile, current_idx, correct_idx):
        self.tile = tile
        self.current_idx = current_idx
        self.correct_idx = correct_idx

    def set_current_idx(self, idx):
        self.current_idx = idx


