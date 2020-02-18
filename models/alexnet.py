import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import dcm


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, module=None):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pool1
            nn.Conv2d(24, 96, 3, padding=1),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pool2
            nn.Conv2d(96, 192, 3, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 96, 3, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pool3
        )

        self.module = Identity()
        if module == "dcm":
            self.module = dcm.IntegratedDCM2d(96, 96, (1, 3))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(96 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.module(x)
        x = x.view(x.size(0), 96 * 4 * 4)  # 寮€濮嬪叏杩炴帴灞傜殑璁＄畻
        x = self.classifier(x)
        return x


def getDefaultAlexNet(module=None):
    return AlexNet(module=module)
