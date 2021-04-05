
import gym
from puzzle_gym.envs.puzzle_env import PuzzleEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
import glob
import numpy as np
from PIL import Image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from custom_cnn import CustomCNN

CHANNEL_NUM = 3
IMAGE_DIR = '../mario'
IMG_SIZE = (33, 33)

def load_data(source):
    
    if source == "dataset":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST('./MNIST/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST('./MNIST/', train=False, transform=transform, download=True)


        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        train_dataset_array = next(iter(train_loader))[0].numpy()
        test_dataset_array = next(iter(test_loader))[0].numpy()
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
            train_dataset_array = dataset_array[ : int(0.8 * array_size)]
            test_dataset_array = dataset_array[int(0.8 * array_size):]

    return train_dataset_array, test_dataset_array

def train():
    train_images, test_images = load_data("local")

    env = Monitor(PuzzleEnv(images=train_images,
                            img_size=IMG_SIZE, puzzle_size=(3, 3), puzzle_type="switch", dist_type="manhattan", penalty_for_step=-0.2,
                            reward_for_completiton=20, positive_reward_coefficient=1.0))

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )

    model = DQN('CnnPolicy', env, policy_kwargs=policy_kwargs,
                verbose=1, learning_rate=0.001, seed=42)
    model.learn(total_timesteps=150000)

    test_env = Monitor(PuzzleEnv(images=test_images,
                            img_size=IMG_SIZE, puzzle_size=(3, 3), puzzle_type="switch", dist_type="manhattan", penalty_for_step=-0.2,
                            reward_for_completiton=20, positive_reward_coefficient=1.0))

    obs = test_env.reset()
    done = False
    rews = []
    i = 0
    while not done:
        i += 1
        action, _states = model.predict(obs)
        obs, rewards, done, info = test_env.step(action)
        rews.append(rewards)
        print(rewards)
        test_env.render()
    print(rews, i, sum(rews))


if __name__ == "main":
    train()