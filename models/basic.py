import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return x
