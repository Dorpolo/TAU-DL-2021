import os
import copy
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils import load_CIFAR_batch, load_CIFAR10, CIFAR10Data, count_parameters
from logger import logger

BATCH_SIZE = 16
NUM_CLASSES = 10
NUM_EPOCHS = 20
train_val_split = 0.1

CIFAR_DIR: str = f'{os.getcwd()}/data/cifar-10-batches-py'
data = load_CIFAR10(CIFAR_DIR)

train_data_size = len(data['X_train'])
training_index = int((1-train_val_split)*train_data_size)

data['X_val'] = data['X_train'][training_index:]
data['y_val'] = data['y_train'][training_index:]
data['X_train'] = data['X_train'][:training_index]
data['y_train'] = data['y_train'][:training_index]

data_transform = transforms.ToTensor()

train_dataset: CIFAR10Data = CIFAR10Data(images=data['X_train'], labels=data['y_train'], transform=data_transform)
val_dataset: CIFAR10Data = CIFAR10Data(images=data['X_val'],  labels=data['y_val'], transform=data_transform)
test_dataset: CIFAR10Data = CIFAR10Data(images=data['X_test'],  labels=data['y_test'], transform=data_transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

data_loaders: Dict[str, DataLoader] = {
    'train': train_dataloader,
    'val': val_dataloader,
    'test': test_dataloader
}


class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flt = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, NUM_CLASSES)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x: DataLoader) -> nn.Linear:
        flattened = self.flt(x)
        fc1 = self.drop(self.relu(self.fc1(flattened)))
        fc2 = self.drop(self.relu(self.fc2(fc1)))
        fc3 = self.drop(self.relu(self.fc3(fc2)))
        return self.fc4(fc3)


def feed_forward(model: nn.Module,
                 validation_type: str,
                 data_loaders: Dict[str, DataLoader],
                 history: Dict[str, List[int]],
                 num_batches: int):
    """
    :param model: nn.Module
    :param validation_type: one of {'train', 'validation'}
    :param data_loaders: a torched data loaders map.
    :param history: keep KPIs
    :param num_batches: number of batches
    :return:
    """
    assert validation_type in ['train', 'val'], "validation_type should be one of 'train', 'val'"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    running_loss = 0.0
    running_acc = 0.0
    samples = 0
    best_model = copy.deepcopy(model)
    dst_path_best_model = 'fully_connected_best_model.pth'

    model.train() if validation_type == 'train' else model.eval()

    best_fc_val_acc = 0.0

    for i, data_batch in enumerate(data_loaders[validation_type]):
        inputs, labels = data_batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if validation_type == 'train':
            loss.backward()
            optimizer.step()

        samples += labels.size(0)
        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        running_acc += (predicted == labels).sum().item()

        if validation_type == 'train':
            if i % 100 == 0:
                logger.info(f'Batch {i}/{num_batches}: loss: {str(running_loss / samples)[:6]}, acc: { str(running_acc / samples)[:6]}')

        epoch_loss = float(running_loss)/samples
        epoch_acc = float(running_acc)/samples
        history[f"{validation_type}_acc"].append(epoch_acc)
        history[f"{validation_type}_loss"].append(epoch_loss)

        logger.info(f'Epoch {str(epoch)}:  {validation_type} loss: {str(epoch_loss)[:6]}, {validation_type} acc: {str(epoch_acc)[:6]}')

        if validation_type == 'val':
            if epoch_acc > best_fc_val_acc:
                logger.info(f'Saving model val acc improved from {str(best_fc_val_acc)[:6]} to {str(epoch_acc)[:6]}')
                best_fc_val_acc = epoch_acc
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), dst_path_best_model)
            else:
                logger.info(f'val acc did not improve from {str(best_fc_val_acc)[:6]}')


if __name__ == '__main__':
    fully_connected_model = FCNet()
    logger.info(f'Number of parameters for FC model: {count_parameters(fully_connected_model)}')

    kpi_keys = ['train_acc', 'train_loss', 'val_acc', 'val_loss']

    history: Dict[str, List[int]] = {
        key: [] for key in kpi_keys
    }

    num_batches = len(train_dataloader)

    for epoch in range(NUM_EPOCHS):
        logger.info(f'Epoch {epoch}')

        feed_forward(model=fully_connected_model,
                     validation_type='train',
                     data_loaders=data_loaders,
                     history=history,
                     num_batches=num_batches)

        feed_forward(model=fully_connected_model,
                     validation_type='val',
                     data_loaders=data_loaders,
                     history=history,
                     num_batches=num_batches)

