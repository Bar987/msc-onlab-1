import gym
from gym import error, spaces, utils
from gym.utils import seeding
from PIL import Image
import numpy as np
import glob
from math import prod, sqrt
import matplotlib.pyplot as plt

CHANNEL_NUM = 3


class PuzzleEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, images=None, img_size=(100, 100), puzzle_size=(3, 3), puzzle_type="slide", dist_type="manhattan", penalty_for_step=-0.1, reward_for_completiton=1, positive_reward_coefficient=1.0):

        self.puzzle_size = puzzle_size

        self.tile_size = min(
            img_size[0] // puzzle_size[0], img_size[1] // puzzle_size[1])

        self.puzzle_type = puzzle_type

        '''
        A kép méretét a csempék számának és a csempék szélességének szorzatára változtatjuk.
        '''
        self.img_size = (
            self.puzzle_size[0] * self.tile_size, self.puzzle_size[1] * self.tile_size)

        self.images = np.array(images)

        if self.puzzle_type == "slide":
            '''
            Az action space 4 diszkrét akcióból áll (balra, fel, jobbra, le)
            '''
            self.action_space = spaces.Discrete(4)
        else:
            '''
            Az action space a csempék sorainak számainak és az oszlopok száma -1 -nek a szorzata,
            plusz az oszlopok és a sorok száma -1 szorzata.
            '''
            self.action_space = spaces.Discrete(self.puzzle_size[0] * (self.puzzle_size[1] - 1)
                                                + self.puzzle_size[1] * (self.puzzle_size[0] - 1))
        '''
        Az observation space a puzzle által aktuálisan ábrázolt kép.
        '''
        self.observation_space = spaces.Box(low=0, high=255, shape=(
            CHANNEL_NUM, self.img_size[0], self.img_size[1]), dtype=np.uint8)

        self.dist_type = dist_type
        self.penalty_for_step = penalty_for_step
        self.reward_for_completiton = reward_for_completiton
        self.positive_reward_coefficient = positive_reward_coefficient

        self.reset()
        self.seed()

    def step(self, action):
        self.last_reward = self.penalty_for_step

        if self.puzzle_type == "slide":
            reward = self._swap_elements([action])
        else:
            temp = self.puzzle_size[0] * (self.puzzle_size[1]-1)
            if action < temp:
                '''
                Oszlopok között cserélünk.
                '''
                divident = action // (self.puzzle_size[1]-1)
                stride = action % (self.puzzle_size[1]-1)
                idx = divident * self.puzzle_size[1] + stride
                reward = self._swap_elements([idx, idx + 1])
            else:
                '''
                Sorok között cserélünk.
                '''
                idx = action % temp
                reward = self._swap_elements([idx, idx + self.puzzle_size[1]])

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
        try:
            self.image = np.random.default_rng().choice(
                self.images, replace=False, shuffle=False)
        except AttributeError:
            pass
        self._init_puzzle()
        self.last_reward = 0
        return self._get_observation_from_puzzle()

    def render(self, mode='human'):

        if mode == "human":
            for i in range(self.puzzle_size[0]):
                for j in range(self.puzzle_size[1]):
                    idx = i * self.puzzle_size[1] + j
                    print('{0:2d}'.format(
                        self.puzzle[idx].correct_idx), end="  ")
                print('\n', end="")
        else:
            for i in range(self.puzzle_size[0]):
                for j in range(self.puzzle_size[1]*self.tile_size):
                    idx = i * self.puzzle_size[1] + j % self.puzzle_size[1]
                    k = j // self.puzzle_size[1]
                    for m in range(self.tile_size):
                        print('{0:2f}'.format(
                            self.puzzle[idx].tile[k][m][0]), end="  ")
                    if j % self.puzzle_size[1] == self.puzzle_size[1] - 1:
                        print('\n', end="")
                    else:
                        print('\t', end="")
                print('\n', end="")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _init_puzzle(self):
        try:
            puzzle = self.image
        except AttributeError:
            max_num = prod(self.img_size)
            puzzle = np.arange(max_num).repeat(
                CHANNEL_NUM).reshape(self.img_size[0], self.img_size[1], CHANNEL_NUM)

            puzzle = 255 * puzzle / max_num

        '''
        Megfelelő számú darabra vágjuk a képet.
        '''
        puzzle = np.array(np.hsplit(puzzle, self.puzzle_size[1]))
        puzzle = np.array(np.hsplit(puzzle, self.puzzle_size[0]))

        self.puzzle = []

        for i, row in enumerate(puzzle):
            for j, tile in enumerate(row):
                idx = i * self.puzzle_size[1] + j
                self.puzzle.append(Tile(tile, idx, idx))

        '''
        Ha tologatós puzzle-ről van szó, akkor az utolsó csempét kinullázzuk.
        '''
        if self.puzzle_type == "slide":
            idx = self.puzzle_size[0] * self.puzzle_size[1] - 1
            self.puzzle[-1] = Tile(np.zeros(shape=(self.tile_size,
                                                   self.tile_size, CHANNEL_NUM), dtype=np.uint8), idx, idx)
            self._shuffle_puzzle()
        else:
            np.random.shuffle(self.puzzle)

        for i, tile in enumerate(self.puzzle):
            tile.set_current_idx(i)

    '''
    Megkeveri a tologatós puzzle-t, ha nem megoldható permutációt kap, akkor a 0 és 1 correct_idx-ű
    csempét megcseréli, ezzel a permutáció paritását megváltoztatva.
    '''

    def _shuffle_puzzle(self):
        np.random.shuffle(self.puzzle)
        for i, tile in enumerate(self.puzzle):
            tile.set_current_idx(i)
        if not self._check_if_solvable():
            zero_indexed = [
                tile for tile in self.puzzle if tile.correct_idx == 0][0]
            one_indexed = [
                tile for tile in self.puzzle if tile.correct_idx == 1][0]
            self.puzzle[zero_indexed.current_idx] = one_indexed
            self.puzzle[one_indexed.current_idx] = zero_indexed

    '''
    Csak azok a permutációk megoldhatóak, ahol az adott correct_idx-ű tile "előtti" (bal-fentről nézve)
    nagyobb correct_idx-ű tile-ok száma minden csempére összegezve páros.
    '''

    def _check_if_solvable(self):
        parity = 0

        for i, tile in enumerate(self.puzzle):
            for prev_tile in self.puzzle[:i]:
                if prev_tile.correct_idx > tile.correct_idx:
                    parity += 1

        return parity % 2 == 0

    '''
    Az akció következtében megtörténő csempecserét implementálja.
    '''

    def _swap_elements(self, action):

        if len(action) == 1:
            idx = self.puzzle_size[0] * self.puzzle_size[1] - 1
            space = [
                tile for tile in self.puzzle if tile.correct_idx == idx][0]
            action = [space.current_idx, self._map_action_to_index(
                action[0], space.current_idx)]

        score_before = self._calculate_reward(action[0], action[1])
        temp = self.puzzle[action[0]]
        self.puzzle[action[0]] = self.puzzle[action[1]]
        self.puzzle[action[1]] = temp

        for i, tile in enumerate(self.puzzle):
            tile.set_current_idx(i)

        score_after = self._calculate_reward(action[1], action[0])
        return score_before - score_after

    '''
    A különböző akciók függvényében vissazadja a befolyásolt csempe indexét.
    0: Az üres helyre jobbról érkezik csempe.
    1: Az üres helyre lentről érkezik csempe.
    2: Az üres helyre balról érkezik csempe.
    3: Az üres helyre fentről érkezik csempe.
    '''

    def _map_action_to_index(self, action, index):
        max_idx = idx = self.puzzle_size[0] * self.puzzle_size[1] - 1
        result = index

        if action == 0:
            result = index if index % self.puzzle_size[1] == self.puzzle_size[1] - \
                1 else index + 1
        elif action == 1:
            result = index if index // self.puzzle_size[1] == self.puzzle_size[0] - 1 else index + \
                self.puzzle_size[1]
        elif action == 2:
            result = index if index % self.puzzle_size[1] == 0 else index - 1
        elif action == 3:
            result = index if index // self.puzzle_size[1] == 0 else index - \
                self.puzzle_size[1]
        if result > max_idx or result < 0:
            return index
        return result

    '''
    Ellenőrzi, hogy a puzzle megoldásra került-e
    '''

    def _is_done(self):
        for i, tile in enumerate(self.puzzle):
            if i != tile.correct_idx:
                return False
        return True

    '''
    Kiszámolja a jutalmat, manhattan, vagy sima ponttávolság alapján.
    '''

    def _calculate_reward(self, idx1, idx2):
        res = 0

        if self.puzzle_type == "slide":
            idx_list = [idx2]
        else:
            idx_list = [idx1, idx2]

        for i in idx_list:
            if self.dist_type == "manhattan":
                correct_coord = (
                    self.puzzle[i].correct_idx // self.puzzle_size[1], self.puzzle[i].correct_idx % self.puzzle_size[1])
                current_coord = (
                    i // self.puzzle_size[1], i % self.puzzle_size[1])
                distance = abs(
                    correct_coord[0] - current_coord[0]) + abs(correct_coord[1] - current_coord[1])
                res += distance / 2

            else:
                correct_coord = (
                    self.puzzle[i].correct_idx // self.puzzle_size[1], self.puzzle[i].correct_idx % self.puzzle_size[1])
                current_coord = (
                    i // self.puzzle_size[1], i % self.puzzle_size[1])
                distance = self._calculate_point_dist(
                    correct_coord, current_coord)
                res += distance

        return res

    def _calculate_point_dist(self, p1, p2):
        return sqrt(
            ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

    '''
    A csempéket tartalmazó listát leképezi a csempékből álló képre, amit a modell feldolgoz.
    '''

    def _get_observation_from_puzzle(self):
        observation = np.zeros(shape=(
            self.img_size[0], self.img_size[1], CHANNEL_NUM))

        for idx, puzzle in enumerate(self.puzzle):
            j = idx % self.puzzle_size[1]
            i = idx // self.puzzle_size[1]
            observation[i * self.tile_size: (i+1) * self.tile_size, j * self.tile_size: (
                j + 1) * self.tile_size] = puzzle.tile

        return np.moveaxis(observation, -1, 0)


class Tile:
    def __init__(self, tile, current_idx, correct_idx):
        self.tile = tile
        self.current_idx = current_idx
        self.correct_idx = correct_idx

    def set_current_idx(self, idx):
        self.current_idx = idx


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
