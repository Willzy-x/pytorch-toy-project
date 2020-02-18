import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import dcm


class DcmNet(nn.Module):

    def __init__(self, num_class=10):
        super(DcmNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            dcm.IntegratedDCM2d(3, self.inchannel),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU()
        )

        self.layer1 = self.make_layer(64, 2, [3], downsample=False)
        self.layer2 = self.make_layer(128, 2, [3], downsample=True)
        self.layer3 = self.make_layer(256, 2, [3], downsample=True)
        self.layer4 = self.make_layer(512, 2, [3], True)
        self.fc = nn.Linear(512, num_class)

    def make_layer(self, channels, num_of_blocks, ks=(1, 3, 5), downsample=False):
        downsamples = [downsample] + [False] * (num_of_blocks - 1)
        layers = []
        for ds in downsamples:
            layers.append(dcm.ResidualIntegratedDCM2d(self.inchannel, channels, ks, ds))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def getDefaultDCMNet():
    return DcmNet()


if __name__ == '__main__':
    x = torch.randn([2, 3, 32, 32])
    dcm = getDefaultDCMNet()
    y = dcm(x)
    print(y.size())