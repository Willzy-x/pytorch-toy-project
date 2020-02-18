import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import dcm, resblock


class DcmResNet(nn.Module):
    
    def __init__(self, ks=(1, 3), num_class=10):
        super(DcmResNet, self).__init__()
        # self.arg = arg
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU()
        )

        self.layer1 = DcmResNet.make_layer(64, resblock.ResidualBlock, 64, 2, stride=1)
        self.layer2 = DcmResNet.make_layer(64, resblock.ResidualBlock, 128, 2, stride=2)
        self.layer3 = DcmResNet.make_layer(128, resblock.ResidualBlock, 256, 2, stride=2)
        self.layer4 = DcmResNet.make_dcms(256, ks, 256)
        self.trans = nn.Conv2d(256 * (len(ks) + 1), 256, kernel_size=1)
        self.layer5 = DcmResNet.make_layer(256, resblock.ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_class)

    @staticmethod
    def make_layer(inchannel, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides = [1,1]
        layers = []
        for stride in strides:
            layers.append(block(inchannel, channels, stride))
            inchannel = channels
        return nn.Sequential(*layers)

    @staticmethod
    def make_dcms(inchannel, ks, outchannel):
        layer = nn.ModuleList()
        for k in ks:
            layer.append(dcm.DynamicContextModule2d(inchannel, outchannel, k))
        return layer

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        outs = []
        for module in self.layer4:
            outs.append(module(out))
        outs.append(out)
        out = self.trans(torch.cat(outs, dim=1))
        out = self.layer5(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def getDefaultDcmResNet():
    return DcmResNet()


if __name__ == '__main__':
    x = torch.randn([2, 3, 32, 32])
    drnet = DcmResNet()
    y = drnet(x)
    print(y.size())