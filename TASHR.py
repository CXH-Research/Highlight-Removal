import torch
import torch.nn as nn
import torch.nn.functional as F


class GenConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding='SAME', activation=nn.LeakyReLU()):
        super(GenConv, self).__init__()
        if padding == 'SAME':
            pad = kernel_size // 2
            self.pad_layer = None
        elif padding in ['SYMMETRIC', 'REFLECT']:
            pad = int(dilation * (kernel_size - 1) / 2)
            if padding == 'SYMMETRIC':
                self.pad_layer = nn.ReflectionPad2d(pad)
            else:
                self.pad_layer = nn.ReplicationPad2d(pad)
            pad = 0
        else:
            self.pad_layer = None
            pad = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=pad, dilation=dilation)
        self.activation = activation

    def forward(self, x):
        if self.pad_layer is not None:
            x = self.pad_layer(x)
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class GenDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, padding='SAME', activation=nn.LeakyReLU()):
        super(GenDeconv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = GenConv(in_channels, out_channels, kernel_size=3,
                            stride=1, padding=padding, activation=activation)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, activation=nn.LeakyReLU(), padding='SAME'):
        super(ResBlock, self).__init__()
        if padding == 'SAME':
            pad = 1
        else:
            pad = 0

        self.conv1 = nn.Conv2d(channels, channels // 4,
                               kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channels // 4, channels //
                               4, kernel_size=3, stride=1, padding=pad)
        self.conv3 = nn.Conv2d(channels // 4, channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(channels)
        self.activation = activation

    def forward(self, x):
        residual = x
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        out = self.conv3(out)
        out = out + residual
        out = self.activation(self.bn(out))
        return out


class TASHR(nn.Module):
    def __init__(self):
        super(TASHR, self).__init__()
        cnum = 32

        # Stage 1 - Initial Layers
        self.t1_conv1 = GenConv(4, cnum//2, 3, 1)
        self.t1_conv2 = GenConv(cnum//2, cnum//2, 3, 1)
        self.t1_conv3 = GenConv(cnum//2, cnum, 3, 2)
        self.t1_conv4 = GenConv(cnum, cnum, 3, 1)
        self.t1_conv5 = GenConv(cnum, cnum, 3, 1)
        self.t1_conv6 = GenConv(cnum, 2 * cnum, 3, 2)
        self.t1_conv7 = GenConv(2 * cnum, 2 * cnum, 3, 1)
        self.t1_conv8 = GenConv(2 * cnum, 2 * cnum, 3, 1)
        self.t1_conv9 = GenConv(2 * cnum, 4 * cnum, 3, 2)
        self.t1_conv10 = GenConv(4 * cnum, 4 * cnum, 3, 1)

        self.t1_conv11 = GenDeconv(4 * cnum, 2 * cnum)
        self.t1_conv12 = GenConv(4 * cnum, 2 * cnum, 3, 1)
        self.t1_conv13 = GenConv(2 * cnum, 2 * cnum, 3, 1)
        self.t1_conv14 = GenConv(2 * cnum, 2 * cnum, 3, 1)
        self.t1_conv15 = GenDeconv(2 * cnum, cnum)
        self.t1_conv16 = GenConv(2 * cnum, cnum, 3, 1)
        self.t1_conv17 = GenConv(cnum, cnum, 3, 1)
        self.t1_conv18 = GenConv(cnum, cnum, 3, 1)
        self.t1_conv19 = GenDeconv(cnum, cnum // 2)
        self.t1_conv20 = GenConv(cnum, cnum // 2, 3, 1)
        self.x_score1 = GenConv(cnum // 2, 1, 3, 1, activation=None)

        # Stage 1 - Main Network
        self.conv1 = GenConv(5, cnum, 5, 1)
        self.conv2_downsample = GenConv(cnum, 2 * cnum, 3, 2)
        self.conv3 = GenConv(2 * cnum, 2 * cnum, 3, 1)
        self.conv4_downsample = GenConv(2 * cnum, 4 * cnum, 3, 2)
        self.conv5 = GenConv(4 * cnum, 4 * cnum, 3, 1)
        self.conv6 = GenConv(4 * cnum, 4 * cnum, 3, 1)
        self.s1_resblock1 = ResBlock(4 * cnum)
        self.s1_resblock2 = ResBlock(4 * cnum)
        self.s1_resblock3 = ResBlock(4 * cnum)
        self.s1_resblock4 = ResBlock(4 * cnum)
        self.conv11 = GenConv(4 * cnum, 4 * cnum, 3, 1)
        self.conv12 = GenConv(8 * cnum, 4 * cnum, 3, 1)
        self.conv13_upsample = GenDeconv(8 * cnum, 2 * cnum)
        self.conv14 = GenConv(4 * cnum, 2 * cnum, 3, 1)
        self.conv15_upsample = GenDeconv(4 * cnum, cnum)
        self.conv16 = GenConv(2 * cnum, cnum // 2, 3, 1)
        self.conv17 = GenConv(cnum // 2, 3, 3, 1, activation=None)

    def forward(self, x):
        # x: [B, C, H, W]
        xin = x
        ones_x = torch.ones_like(x[:, :1, :, :])
        x = torch.cat([x, ones_x], dim=1)

        # Stage 1 - Initial Layers
        t1_conv1 = self.t1_conv1(x)
        t1_conv2 = self.t1_conv2(t1_conv1)
        t1_conv3 = self.t1_conv3(t1_conv2)
        t1_conv4 = self.t1_conv4(t1_conv3)
        t1_conv5 = self.t1_conv5(t1_conv4)
        t1_conv6 = self.t1_conv6(t1_conv5)
        t1_conv7 = self.t1_conv7(t1_conv6)
        t1_conv8 = self.t1_conv8(t1_conv7)
        t1_conv9 = self.t1_conv9(t1_conv8)
        t1_conv10 = self.t1_conv10(t1_conv9)
        t1_conv11 = self.t1_conv11(t1_conv10)
        t1_conv11 = torch.cat([t1_conv8, t1_conv11], dim=1)
        t1_conv12 = self.t1_conv12(t1_conv11)
        t1_conv13 = self.t1_conv13(t1_conv12)
        t1_conv14 = self.t1_conv14(t1_conv13)
        t1_conv15 = self.t1_conv15(t1_conv14)
        t1_conv15 = torch.cat([t1_conv5, t1_conv15], dim=1)
        t1_conv16 = self.t1_conv16(t1_conv15)
        t1_conv17 = self.t1_conv17(t1_conv16)
        t1_conv18 = self.t1_conv18(t1_conv17)
        t1_conv19 = self.t1_conv19(t1_conv18)
        t1_conv19 = torch.cat([t1_conv2, t1_conv19], dim=1)
        t1_conv20 = self.t1_conv20(t1_conv19)
        x_score1 = self.x_score1(t1_conv20)

        # Stage 1 - Main Network
        xnow = torch.cat([xin, ones_x, x_score1], dim=1)
        s1_conv1 = self.conv1(xnow)
        s1_conv2 = self.conv2_downsample(s1_conv1)
        s1_conv3 = self.conv3(s1_conv2)
        s1_conv4 = self.conv4_downsample(s1_conv3)
        s1_conv5 = self.conv5(s1_conv4)
        s1_conv6 = self.conv6(s1_conv5)
        s1_conv7 = self.s1_resblock1(s1_conv6)
        s1_conv8 = self.s1_resblock2(s1_conv7)
        s1_conv9 = self.s1_resblock3(s1_conv8)
        s1_conv10 = self.s1_resblock4(s1_conv9)
        s1_conv11 = self.conv11(s1_conv10)
        s1_conv11 = torch.cat([s1_conv6, s1_conv11], dim=1)
        s1_conv12 = self.conv12(s1_conv11)
        s1_conv12 = torch.cat([s1_conv5, s1_conv12], dim=1)
        s1_conv13 = self.conv13_upsample(s1_conv12)
        s1_conv13 = torch.cat([s1_conv3, s1_conv13], dim=1)
        s1_conv14 = self.conv14(s1_conv13)
        s1_conv14 = torch.cat([s1_conv2, s1_conv14], dim=1)
        s1_conv15 = self.conv15_upsample(s1_conv14)
        s1_conv15 = torch.cat([s1_conv1, s1_conv15], dim=1)
        s1_conv16 = self.conv16(s1_conv15)
        s1_conv17 = self.conv17(s1_conv16)
        x_stage1 = torch.clamp(s1_conv17, -1., 1.)

        return x_stage1


if __name__ == '__main__':
    model = TASHR()
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(y.shape)