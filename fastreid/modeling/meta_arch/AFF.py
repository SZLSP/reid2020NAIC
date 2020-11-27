import torch
from torch import nn

class MS_CAM(nn.Module):
    def __init__(self, C, H, W, r):
        """
        MS_CAM is a module of AFF.
        Args:
            C: Channel
            H: Height
            W: Width
            r: channel reduction ratio. The channel will be reduced to C/r and back to C.
        """
        super(MS_CAM, self).__init__()
        self.get_global_feature = nn.Sequential(
        )
        self.get_local_feature = nn.Sequential(
        )

        interdim = max(C // r, 1)

        self.globalavgpool = nn.AvgPool2d((H, W))
        self.PWConv11 = nn.Conv2d(C, interdim, 1, 1, 0, bias=False)
        self.bn11 = nn.BatchNorm2d(interdim)
        self.PWConv12 = nn.Conv2d(interdim, C, 1, 1, 0, bias=False)
        self.bn12 = nn.BatchNorm2d(C)

        self.PWConv21 = nn.Conv2d(C, interdim, 1, 1, 0, bias=False)
        self.bn21 = nn.BatchNorm2d(interdim)
        self.relu = nn.ReLU(inplace=True)
        self.PWConv22 = nn.Conv2d(interdim, C, 1, 1, 0, bias=False)
        self.bn22 = nn.BatchNorm2d(C)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        global_feature = self.globalavgpool(x)
        global_feature = self.PWConv11(global_feature)
        global_feature = self.bn11(global_feature)
        global_feature = self.relu(global_feature)
        global_feature = self.PWConv12(global_feature)
        global_feature = self.bn12(global_feature)

        local_feature = self.PWConv21(x)
        local_feature = self.bn21(local_feature)
        local_feature = self.relu(local_feature)
        local_feature = self.PWConv22(local_feature)
        local_feature = self.bn22(local_feature)

        x2 = self.sigmoid(global_feature + local_feature)

        return x * x2

class AFF(nn.Module):
    def __init__(self, C, H, W, r):
        super(AFF, self).__init__()
        self.MS_CAM = MS_CAM(C, H, W, r)

    def forward(self, X, Y):
        assert X.shape == Y.shape, "Input of AFF(X and Y) should be the same shape"

        M = self.MS_CAM(X + Y)
        Z = M * X + (1 - M) * Y

        return Z

class iAFF(nn.Module):
    def __init__(self, C, H, W, r):
        super(iAFF, self).__init__()
        self.AFF = AFF(C, H, W, r)
        self.MS_CAM2 = MS_CAM(C, H, W, r)

    def forward(self, X, Y):
        assert X.shape == Y.shape, "Input of AFF(X and Y) should be the same shape"
        M = self.AFF(X, Y)
        M = self.MS_CAM2(M)
        Z = M * X + (1 - M) * Y

        return Z

if __name__ == '__main__':
    i = torch.randn((50, 10, 5, 3))
    m = MS_CAM(10, 5, 3, 2)
    out0 = m(i)

    a = AFF(10, 5, 3, 2)
    b = iAFF(10, 5, 3, 2)
    j = torch.randn((50, 10, 5, 3))
    out1 = a(i, i)
    out2 = b(i, i)

    pass

