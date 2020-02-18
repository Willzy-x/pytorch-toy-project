import torch
import torch.nn as nn
import torch.nn.functional as F


def make_dcms(inchannel, ks, outchannel):
    layer = nn.ModuleList()
    for k in ks:
        layer.append(DynamicContextModule2d(inchannel, outchannel, k))
    return layer


class DynamicContextModule2d(nn.Module):

    def __init__(self, inChan, outChan, k):
        super(DynamicContextModule2d, self).__init__()
        self.k = k
        self.upBranch = nn.Conv2d(inChan, outChan, kernel_size=1)
        self.downBranch = nn.Sequential(
            nn.AdaptiveAvgPool2d(k),
            nn.Conv2d(inChan, outChan, kernel_size=1)
        )
        self.final = nn.Conv2d(outChan, outChan, kernel_size=1)
        self.groups = outChan

    def forward(self, x):
        N = x.size()[0]
        up = torch.split(self.upBranch(x), 1, 0)
        down = torch.split(self.downBranch(x), 1, 0)
        out = []
        for i in range(N):
            temp = F.conv2d(input=up[i], weight=down[i].squeeze(0).unsqueeze(1),
                            padding=(self.k - 1) // 2, groups=self.groups)
            out.append(temp)
        out = torch.cat(out, dim=0)
        out = self.final(out)
        return out


class DynamicContextModule3d(nn.Module):

    def __init__(self, inChan, outChan, k):
        super(DynamicContextModule3d, self).__init__()
        self.k = k
        self.upBranch = nn.Conv3d(inChan, outChan, kernel_size=1)
        self.downBranch = nn.Sequential(
            nn.AdaptiveAvgPool3d(k),
            nn.Conv3d(inChan, outChan, kernel_size=1)
        )
        self.final = nn.Conv3d(outChan, outChan, kernel_size=1)
        self.groups = outChan

    def forward(self, x):
        N = x.size()[0]
        up = torch.split(self.upBranch(x), 1, 0)
        down = torch.split(self.downBranch(x), 1, 0)
        out = []
        for i in range(N):
            temp = F.conv3d(input=up[i], weight=down[i].squeeze(0).unsqueeze(1),
                            padding=(self.k - 1) // 2, groups=self.groups)
            out.append(temp)
        out = torch.cat(out, dim=0)
        out = self.final(out)
        return out


class IntegratedDCM2d(nn.Module):

    def __init__(self, inchannel, outchannel, ks=(1, 3, 5)):
        super(IntegratedDCM2d, self).__init__()
        self.layer = nn.ModuleList()
        for k in ks:
            self.layer.append(DynamicContextModule2d(inchannel, outchannel, k))
        self.outTrans = nn.Conv2d(outchannel * len(ks) + inchannel, outchannel, kernel_size=1)

    def forward(self, x):
        outs = []
        for module in self.layer:
            outs.append(module(x))
        outs.append(x)
        out = torch.cat(outs, dim=1)
        out = self.outTrans(out)
        return out


class ResidualIntegratedDCM2d(nn.Module):

    def __init__(self, inchannel, outchannel, ks=(1, 3, 5), downsample=False):
        super(ResidualIntegratedDCM2d, self).__init__()
        self.downsample = downsample
        self.left = nn.Sequential(
            IntegratedDCM2d(inchannel, outchannel, ks),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            IntegratedDCM2d(outchannel, outchannel, ks),
            nn.BatchNorm2d(outchannel)
        )

        if self.downsample:
            step = 2
        else:
            step = 1
        self.shortcut = nn.Sequential()
        if inchannel != outchannel or self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=step, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        x_right = x
        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        out = self.left(x)
        out += self.shortcut(x_right)
        out = F.relu(out)
        return out


if __name__ == '__main__':
    x = torch.randn([2, 4, 32, 32])
    dcm = ResidualIntegratedDCM2d(4, 2, downsample=True)
    y = dcm(x)
    print(y.size())