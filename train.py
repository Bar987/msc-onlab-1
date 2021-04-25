
import gym
from puzzle_gym.envs.puzzle_env import PuzzleEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import statistics

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from custom_cnn import CustomCNN

OBS_CONF = {"min": 0, "max": 1, "type": np.float32}
CHANNEL_NUM = 1
IMAGE_DIR = '../'
IMG_SIZE = (27, 27)
RATIO = 1/9


def load_data(source):

    if source == "dataset":

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(IMG_SIZE)])
        train_dataset = datasets.MNIST(
            './MNIST/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(
            './MNIST/', train=False, transform=transform, download=True)

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        train_dataset_array = get_usable_images(
            np.moveaxis(next(iter(train_loader))[0].numpy(), 1, -1), RATIO)
        test_dataset_array = get_usable_images(
            np.moveaxis(next(iter(test_loader))[0].numpy(), 1, -1), RATIO)
    else:
        if IMAGE_DIR:
            images = []
            for f in glob.iglob(IMAGE_DIR + "/*"):
                if CHANNEL_NUM == 1:
                    images.append(np.asarray(Image.open(f).resize(IMG_SIZE))[..., :CHANNEL_NUM].reshape(
                        (IMG_SIZE[0], IMG_SIZE[1], CHANNEL_NUM)))
                else:
                    images.append(np.asarray(Image.open(f).resize(IMG_SIZE))[..., :CHANNEL_NUM].reshape(
                        (IMG_SIZE[0], IMG_SIZE[1], CHANNEL_NUM)))
            dataset_array = np.array(images)
            array_size = len(dataset_array)
            train_dataset_array = dataset_array
            test_dataset_array = train_dataset_array

    return train_dataset_array, test_dataset_array


def train():
    train_images, test_images = load_data("dataset")

    env = Monitor(PuzzleEnv(images=train_images,
                            img_size=IMG_SIZE, channel_num=CHANNEL_NUM, puzzle_size=(3, 3), max_step_num=100, puzzle_type="switch", dist_type="manhattan", penalty_for_step=-0.2,
                            reward_for_completiton=20, positive_reward_coefficient=1.0, obs_conf=OBS_CONF))

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs,
                verbose=1, learning_rate=0.0005, seed=42)
    model.learn(total_timesteps=1000000)

    test(model, test_images)


def test(model, test_images):
    test_env = Monitor(PuzzleEnv(images=test_images,
                                 img_size=IMG_SIZE, channel_num=CHANNEL_NUM, puzzle_size=(3, 3), puzzle_type="switch", dist_type="manhattan", penalty_for_step=-0.2,
                                 reward_for_completiton=20, positive_reward_coefficient=1.0, obs_conf=OBS_CONF))

    solutions = []
    rews = []
    steps = []
    sample = len(test_images)
    errors = 0

    for iter in range(sample):
        i = 0
        done = False
        obs = test_env.reset()
        frames = [obs]

        while not done:
            i += 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = test_env.step(action)
            frames.append(obs)
            rews.append(rewards)

            if i == 10000:
                errors += 1
                break

        solutions.append(frames)
        done = False
        print(i, sum(rews), rews)
        rews = []
        steps.append(i)

    print('Average steps taken:  ', sum(steps) / sample)
    print('Median of steps taken: ', statistics.median(steps))
    print('Number of errors: ', errors)
    plt.hist(steps, bins=9)
    plt.savefig('fig.png')


def get_usable_images(arr, threshold):
    good_img_idx = []
    global_tile_ratios = []
    img_ratio = []
    for idx, img in enumerate(arr):
        tile_ratios = []
        puzzle = np.array(np.hsplit(img, 3))
        puzzle = np.array(np.hsplit(puzzle, 3))
        img_ratio.append((img.size-np.count_nonzero(img == 0)) / img.size)
        for i, row in enumerate(puzzle):
            for j, tile in enumerate(row):
                count_black = np.count_nonzero(tile == 0)
                count_white = tile.size - count_black
                tile_ratios.append(count_white / tile.size)

        bad_tiles = [x for x in tile_ratios if x <= threshold]

        if not len(bad_tiles) > 1:
            good_img_idx.append(idx)

        global_tile_ratios += tile_ratios
    print(len(good_img_idx)/len(arr))
    return arr[good_img_idx]


if __name__ == "__main__":
    train()
