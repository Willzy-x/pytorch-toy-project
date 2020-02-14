import torch
import torch.nn as nn
import torch.nn.functional as F


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


if __name__ == '__main__':
    x = torch.randn([2, 4, 16, 16, 16])
    dcm = DynamicContextModule3d(4, 2, 2)
    y = dcm(x)
    print(y.size())