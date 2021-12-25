import os
import logging

import pickle
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


class CIFAR10Data(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = torch.tensor(self.labels[index], dtype=torch.long)
        image = self.images[index].astype('uint8')
        if self.transform:
            image = self.transform(image)
        return image, label


def load_CIFAR_batch(filename):
    with open(filename, "rb") as f:
        datadict = pickle.load(f, encoding="latin1")
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar batches (1-5 + test)"""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return {
        "X_train": Xtr,
        "y_train": Ytr,
        "X_test": Xte,
        "y_test": Yte,
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)