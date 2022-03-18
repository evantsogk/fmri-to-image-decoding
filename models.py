import torch
import torch.nn as nn


def weights_init(m):
    """
    Initializes weights from a normal distribution according to the DCGAN paper.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)


class DeconvBlock(nn.Module):
    """
    Deconvolutional block used by the Generator.
    Better to use kernel as multiple of stride to avoid the checkerboard effect.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DeconvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    The Generator.
    Takes as input a vector with dimension equal to the ROIs dimension.
    """
    def __init__(self, fmri_dim):
        super(Generator, self).__init__()

        self.fc = nn.Linear(fmri_dim, 512*7*7)
        self.model = nn.Sequential(
            DeconvBlock(512, 256, kernel_size=4, stride=2, padding=1),
            DeconvBlock(256, 128, kernel_size=4, stride=2, padding=1),
            DeconvBlock(128, 128, kernel_size=4, stride=2, padding=1),
            DeconvBlock(128, 128, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

        self.apply(weights_init)  # initialize the weights

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 7, 7)
        return self.model(x)


class ConvBlock(nn.Module):
    """
    Convolutional block used by the Discriminator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    """
    The Discriminator.
    """
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=4, stride=2, padding=0),
            ConvBlock(32, 32, kernel_size=4, stride=2, padding=0),
            ConvBlock(32, 64, kernel_size=4, stride=2, padding=0),
            ConvBlock(64, 64, kernel_size=4, stride=2, padding=0),

            nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=0)
        )

        self.apply(weights_init)  # initialize the weights

    def forward(self, x):
        x = self.model(x)
        return x.view(len(x), -1)  # the output is a single value so reduce to one dimension
