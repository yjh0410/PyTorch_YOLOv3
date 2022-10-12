import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import os


model_urls = {
    "darknet53": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet53.pth",
}


__all__ = ['darknet19']


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, c1, c2, k=1, p=0, s=1, d=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, padding=p, stride=s, dilation=d),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class resblock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch//2, k=1),
                Conv_BN_LeakyReLU(ch//2, ch, k=3, p=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet_53(nn.Module):
    """
    DarkNet-53.
    """
    def __init__(self, num_classes=1000):
        super(DarkNet_53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, k=3, p=1),
            Conv_BN_LeakyReLU(32, 64, k=3, p=1, s=2),
            resblock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, k=3, p=1, s=2),
            resblock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, k=3, p=1, s=2),
            resblock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, k=3, p=1, s=2),
            resblock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, k=3, p=1, s=2),
            resblock(1024, nblocks=4)
        )


    def forward(self, x, targets=None):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        output = {
            'c3': c3,
            'c4': c4,
            'c5': c5
        }
        return output


def build_darknet53(pretrained=False):
    # model
    model = DarkNet_53()
    feat_dims = [256, 512, 1024]

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['darknet53']
        # checkpoint state dict
        checkpoint_state_dict = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model, feat_dims


if __name__ == '__main__':
    import time
    model, feats = build_darknet53(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    outputs = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for k in outputs.keys():
        print(outputs[k].shape)
