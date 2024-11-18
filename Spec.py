import torch
import torch.nn as nn
import functools


class ResnetBlock(nn.Module):
    """Define a Resnet block with reflection padding and instance normalization."""

    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """Initialize the Resnet block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero.
            norm_layer          -- normalization layer.
            use_dropout (bool)  -- if use dropout layers.
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero.
            norm_layer          -- normalization layer.
            use_dropout (bool)  -- if use dropout layers.

        Returns:
            conv_block (nn.Sequential) -- sequential convolutional block.
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:  # zero padding
            p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True),
                       norm_layer(dim),
                       nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # Second convolution
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:  # zero padding
            p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)."""
        return x + self.conv_block(x)  # add skip connections


class Spec(nn.Module):
    """Resnet-based generator with reflection padding and instance normalization."""

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        """Construct a Resnet-based generator.

        Parameters:
            input_nc (int)      -- the number of channels in input images.
            output_nc (int)     -- the number of channels in output images.
            ngf (int)           -- the number of filters in the last conv layer.
            n_blocks (int)      -- the number of ResNet blocks.
        """
        super(Spec, self).__init__()
        assert(n_blocks >= 0)
        norm_layer = nn.InstanceNorm2d

        # Initial convolution layers
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Downsampling layers
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type='reflect',
                                  norm_layer=norm_layer, use_dropout=False)]

        # Upsampling layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=True),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward function."""
        return self.model(input)


if __name__ == '__main__':
    model = Spec()
    input = torch.randn(1, 3, 256, 256)
    output = model(input)
    print(output.size())
