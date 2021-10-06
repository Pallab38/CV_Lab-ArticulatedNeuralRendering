import torch
import torch.nn as nn


def down(in_channels, out_channels, with_bn=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2, bias=not with_bn),
        nn.ReLU(inplace=True),
    ]
    if with_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    else:
        layers.append(nn.GroupNorm(8, num_channels=out_channels))
    layers.append(nn.Conv2d(out_channels, out_channels, 1, padding=0, stride=1))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def middle(in_channels, out_channels, with_bn=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, 1, bias=not with_bn),
        nn.ReLU(inplace=True),
    ]
    if with_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    else:
        layers.append(nn.GroupNorm(8, num_channels=out_channels))
    layers.append(nn.Conv2d(out_channels, out_channels, 1))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def up(in_channels, out_channels, with_bn=True):
    layers = [
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, padding=0, stride=2
        ),
        nn.ReLU(inplace=True),
    ]
    if with_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    else:
        layers.append(nn.GroupNorm(8, num_channels=out_channels))
    layers.append(nn.Conv2d(out_channels, out_channels, 2, padding=0, stride=1))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=256, with_bn=True):
        super(UNet, self).__init__()

        # 1024 x 1024 -> 512 x 512
        self.down1 = down(
            in_channels=in_channels, out_channels=mid_channels, with_bn=with_bn
        )

        # 512 x 512 -> 256 x 256
        self.down2 = down(
            in_channels=mid_channels, out_channels=mid_channels * 2, with_bn=with_bn
        )

        # 256 x 256 -> 256 x 256
        self.mid1 = middle(mid_channels * 2, mid_channels * 8)
        self.mid2 = middle(
            mid_channels * 8 + mid_channels * 2, mid_channels * 2, with_bn=with_bn
        )

        # 256 x 256 -> 512 x 512
        self.up2 = up(
            in_channels=mid_channels * 2 + mid_channels * 2,
            out_channels=mid_channels,
            with_bn=with_bn,
        )

        # 512 x 512 -> 1024 x 1024
        self.up1 = up(
            in_channels=mid_channels + mid_channels,
            out_channels=mid_channels // 2,
            with_bn=with_bn,
        )

        self.out = nn.Conv2d(
            in_channels=mid_channels // 2, out_channels=out_channels, kernel_size=1,
        )
        self.sig = nn.Sigmoid()
    def forward(self, im):
        d1 = self.down1(im)  # -> 512x512
        d2 = self.down2(d1)  # -> 256x256
        m1 = self.mid1(d2)  # -> 256x256
        m2 = self.mid2(torch.cat([m1, d2], dim=1))  # -> 256x256
        u1 = self.up2(torch.cat([m2, d2], dim=1))  # -> 512x512
        u2 = self.up1(torch.cat([u1, d1], dim=1))  # -> 1024x1024
        out = self.out(u2)
        img = out[:,0:3,:,:]
        mask = self.sig(out[:,3,:,:])
        return img, mask
