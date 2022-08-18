import numpy as np
import torch.utils.data as Data
from PIL import Image
from transformer import *
import tools


class mnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=0.9, random_seed=1,
                 num_class=10, noise_type='symmetric', anchor=True):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.anchor = anchor
        if anchor:
            original_images = np.load('data/mnist/train_images.npy')
            original_labels = np.load('data/mnist/train_labels.npy')
        else:
            original_images = np.load('data/mnist/mnist_images.npy')
            original_labels = np.load('data/mnist/mnist_labels.npy')

        print(original_images.shape)

        self.train_data, self.val_data, self.train_labels, self.val_labels, self.t = tools.dataset_split(
            original_images,
            original_labels, noise_rate, split_per, random_seed, num_class, noise_type)
        pass

    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class mnist_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/mnist/test_images.npy')
        self.test_labels = np.load('data/mnist/test_labels.npy') - 1  # 0-9

    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)
