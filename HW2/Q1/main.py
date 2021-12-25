import os
import copy
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
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
        fc4 = self.fc4(fc3)
        return fc4


class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding='same')
        self.conv2 = nn.Conv2d(64, 64, 3, padding='same')

        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv4 = nn.Conv2d(128, 128, 3, padding='same')

        self.conv5 = nn.Conv2d(128, 256, 3, padding='same')
        self.conv6 = nn.Conv2d(256, 256, 3, padding='same')

        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

        self.pool = nn.MaxPool2d(2, 2)
        self.flt = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.3)

    def forward(self, x: DataLoader) -> nn.Linear:
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        pool1 = self.pool(conv2)
        drop1_1 = self.drop1(pool1)

        conv3 = self.relu(self.conv3(drop1_1))
        conv4 = self.relu(self.conv4(conv3))
        pool2 = self.pool(conv4)
        drop1_2 = self.drop1(pool2)

        conv5 = self.relu(self.conv5(drop1_2))
        conv6 = self.relu(self.conv6(conv5))
        pool3 = self.pool(conv6)
        drop1_3 = self.drop1(pool3)

        flattened = self.flt(drop1_3)
        fc1 = self.drop2(self.relu(self.fc1(flattened)))
        fc2 = self.fc2(fc1)
        return fc2


if __name__ == '__main__':

    num_batches = len(train_dataloader)

    logger.info(f"{'*'*80}\n{'*'*32} FULLY CONNECTED {'*'*31}\n{'*'*80}")

    fully_connected_model = FCNet()
    logger.info(f'Number of parameters for FC model: {count_parameters(fully_connected_model)}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fully_connected_model.parameters())
    best_fc_val_acc = 0.0
    best_model = copy.deepcopy(fully_connected_model)
    dst_path_best_model = 'fully_connected_best_model.pth'

    kpi_keys = ['train_acc', 'train_loss', 'val_acc', 'val_loss']
    history: Dict[str, List[int]] = {
        key: [] for key in kpi_keys
    }

    for epoch in range(NUM_EPOCHS):
        logger.info(f'Epoch {epoch}')
        for validation_type in ['train', 'val']:
            fully_connected_model.train() if validation_type == 'train' else fully_connected_model.eval()
            best_fc_val_acc = 0.0

            running_loss = 0.0
            running_acc = 0.0
            samples = 0

            for i, data_batch in enumerate(data_loaders[validation_type]):
                inputs, labels = data_batch
                optimizer.zero_grad()
                outputs = fully_connected_model(inputs)
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
                        logger.info(f'Batch {i}/{num_batches}: loss: {str(running_loss / samples)[:6]}, acc: {str(running_acc / samples)[:6]}')

            epoch_loss = float(running_loss) / samples
            epoch_acc = float(running_acc) / samples

            history[f"{validation_type}_acc"].append(epoch_acc)
            history[f"{validation_type}_loss"].append(epoch_loss)

            logger.info(f'Epoch {str(epoch)}:  {validation_type} loss: {str(epoch_loss)[:6]}, {validation_type} acc: {str(epoch_acc)[:6]}')

            if validation_type == 'val':
                if epoch_acc > best_fc_val_acc:
                    logger.info( f'Saving model val acc improved from {str(best_fc_val_acc)[:6]} to {str(epoch_acc)[:6]}')
                    best_fc_val_acc = epoch_acc
                    best_model = copy.deepcopy(fully_connected_model)
                    torch.save(fully_connected_model.state_dict(), dst_path_best_model)
                else:
                    logger.info(f'val acc did not improve from {str(best_fc_val_acc)[:6]}')

    logger.info(f"{'*' * 80}")

    plt.plot(history['train_acc'], c='red')
    plt.plot(history['val_acc'], c='blue')
    plt.legend(["train", "validation"], loc="lower right")
    plt.title('FC accuracy')
    plt.show()
    plt.clf()

    logger.info(f"{'*' * 80}")

    plt.plot(history['train_loss'], c='red')
    plt.plot(history['val_loss'], c='blue')
    plt.legend(["train", "validation"], loc="lower right")
    plt.title('FC loss')
    plt.show()

    logger.info(f"{'*' * 80}")

    logger.info(f"{'*' * 80}\n{'*' * 38} CNN {'*' * 37}\n{'*' * 80}")

    cnn_model = CNNNet()
    logger.info(f'Number of parameters for CNN model: {count_parameters(cnn_model)}')

    cnn_criterion = nn.CrossEntropyLoss()
    cnn_optimizer = optim.Adam(cnn_model.parameters())

    best_cnn_val_acc = 0.0
    best_cnn_model = copy.deepcopy(cnn_model)
    dest_path_best_model = 'CNN_best_model.pth'

    history_cnn: Dict[str, List[int]] = {
        key: [] for key in kpi_keys
    }

    for epoch in range(NUM_EPOCHS):
        logger.info(f'Epoch {epoch}')
        for validation_type in ['train', 'val']:
            cnn_model.train() if validation_type == 'train' else cnn_model.eval()
            running_loss, running_acc, samples = 0.0, 0.0, 0

            for i, data_batch in enumerate(data_loaders[validation_type]):
                inputs, labels = data_batch
                cnn_criterion.zero_grad()
                outputs = cnn_model(inputs)
                cnn_loss = cnn_criterion(outputs, labels)

                if validation_type == 'train':
                    cnn_loss.backward()
                    cnn_optimizer.step()

                samples += labels.size(0)
                running_loss += cnn_loss.item()*labels.size(0)

                _, predicted = torch.max(outputs, 1)
                running_acc += (predicted == labels).sum().item()

                if validation_type == 'train':
                    if i % 100 == 0:
                        logger.info(f'Batch {i}/{num_batches}: loss: {str(running_loss / samples)[:6]}, acc: {str(running_acc / samples)[:6]}')

        epoch_loss = float(running_loss) / samples
        epoch_acc = float(running_acc) / samples

        history_cnn[f"{validation_type}_acc"].append(epoch_acc)
        history_cnn[f"{validation_type}_loss"].append(epoch_loss)

        logger.info(f'Epoch {str(epoch)}:  {validation_type} loss: {str(epoch_loss)[:6]}, {validation_type} acc: {str(epoch_acc)[:6]}')

        if validation_type == 'val':
            if epoch_acc > best_cnn_val_acc:
                logger.info(f'Saving model val acc improved from {str(best_fc_val_acc)[:6]} to {str(epoch_acc)[:6]}')
                best_cnn_val_acc = epoch_acc
                best_cnn_model = copy.deepcopy(cnn_model)
                torch.save(cnn_model.state_dict(), dst_path_best_model)
            else:
                logger.info(f'val acc did not improve from {str(best_cnn_val_acc)[:6]}')

    logger.info(f"{'*' * 80}")

    plt.plot(history_cnn['train_acc'], c='red')
    plt.plot(history_cnn['val_acc'], c='blue')
    plt.legend(["train", "validation"], loc="lower right")
    plt.title('CNN accuracy')
    plt.show()
    plt.clf()

    logger.info(f"{'*' * 80}")

    plt.plot(history_cnn['train_loss'], c='red')
    plt.plot(history_cnn['val_loss'], c='blue')
    plt.legend(["train", "validation"], loc="lower right")
    plt.title('CNN loss')
    plt.show()

    logger.info(f"{'*' * 80}")


