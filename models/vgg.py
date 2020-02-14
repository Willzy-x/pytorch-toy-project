import torch.nn as nn
import math

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # Conv1
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),  # Conv2
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool1
            nn.Conv2d(32, 64, 3, padding=1),  # Conv3
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),  # Conv4
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool2
            nn.Conv2d(64, 128, 3, padding=1),  # Conv5
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),  # Conv6
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),  # Conv7
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool3
            nn.Conv2d(128, 256, 3, padding=1),  # Conv8
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv9
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv10
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool4
            nn.Conv2d(256, 256, 3, padding=1),  # Conv11
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv12
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv13
            nn.ReLU(True),
            # nn.MaxPool2d(2, 2)  # Pool5 是否还需要此池化层，每个通道的数据已经被降为1*1
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * 256, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x