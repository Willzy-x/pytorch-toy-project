from torchvision import datasets as ds
from torchvision import transforms as ts
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from visdom import Visdom
import requests
import os
import numpy as np
from models.resnet import getDefaultResNet
from models.dcmresnet import getDefaultDcmResNet
from models.dcmnet import getDefaultDCMNet
from models.alexnet import getDefaultAlexNet
from train_and_test import train, test

BATCH_SIZE = 256
NUM_OF_WORKERS = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_OF_EPOCHS = 200
VISUAL_EVERY_EPOCH = 50
STEP_SIZE = 70
SAVE_DIR = "./results"
INPUT_CHANNEL = 3
NUM_CLASS = 100

RGB_CYAN = (0, 238, 238)
RGB_MAGENTA = (238, 0, 238)
RGB_FIREBRICK = (255, 48, 48)
RGB_DODGERBLUE = (30, 144, 255)

transform = ts.Compose(
    [
        ts.ToTensor(),
        ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)


def main():
    viz = Visdom(server='http://192.168.1.108', port=8097, env='cifar100')
    assert viz.check_connection()

    train_set = ds.CIFAR100('./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_OF_WORKERS)

    test_set = ds.CIFAR100('./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE * 2, shuffle=True, num_workers=NUM_OF_WORKERS)

    net = getDefaultAlexNet(INPUT_CHANNEL, NUM_CLASS, "dcm").cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ------- Save training data -------------------------------------------------------------
    train_file = open(os.path.join(SAVE_DIR, 'train.csv'), 'w')
    train_file.write('Epoch,Loss,Training Accuracy\n')
    test_file = open(os.path.join(SAVE_DIR, 'test.csv'), 'w')
    test_file.write('Epoch,Loss,Test Accuracy\n ')

    # Starting training process:
    train_loss_plot, test_loss_plot = np.array([]), np.array([])
    train_acc_plot, test_acc_plot = np.array([]), np.array([])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)
    for i in range(1, NUM_OF_EPOCHS + 1):
        print("---------------------------------------------------------------------------------------")
        train_loss, train_acc = train(i, net, optimizer, criterion, train_loader, train_file)
        train_loss_plot = np.append(train_loss_plot, train_loss)
        train_acc_plot = np.append(train_acc_plot, train_acc)

        test_loss, test_acc = test(i, net, criterion, test_loader, test_file)
        test_acc_plot = np.append(test_acc_plot, test_acc)
        test_loss_plot = np.append(test_loss_plot, test_loss)

        scheduler.step(epoch=i)
        print("---------------------------------------------------------------------------------------")

        if i % VISUAL_EVERY_EPOCH == 0:
            x_axis = range(1, i + 1)
            viz.line(
                Y=np.column_stack((train_loss_plot, test_loss_plot)),
                X=np.column_stack((x_axis, x_axis)),
                opts=dict(
                    title="loss plot at epoch (%d)" % i,
                    linecolor=np.row_stack((np.array(RGB_CYAN), np.array(RGB_MAGENTA))),
                    legend=["train loss", "test loss"]
                )
            )

            viz.line(
                Y=np.column_stack((train_acc_plot, test_acc_plot)),
                X=np.column_stack((x_axis, x_axis)),
                opts=dict(
                    title="accuracy plot at epoch (%d)" % i,
                    linecolor=np.row_stack((np.array(RGB_DODGERBLUE), np.array(RGB_FIREBRICK))),
                    legend=["train accuracy", "test accuracy"]
                )
            )


if __name__ == '__main__':
    main()
