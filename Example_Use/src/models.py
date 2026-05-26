# Import -------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import os
import sys

_repo_name = "Example_Use"
_module_path = __file__[: __file__.find(_repo_name) + len(_repo_name)]
sys.path.append(os.path.abspath(_module_path))

from utils.functions import load_config




# Model Configuration -----------------------------------------
config_path = os.path.join(_module_path, "src", "config.yml")
config = load_config(config_path)


# model parameters
img_size_time = config["model"]["img_size"]["time"] 
img_size_freq = config["model"]["img_size"]["freq"]
channels = config["model"]["channels"]  # "number of image channels"
latent_dim = config["model"]["latent_dim"]  # "dimensionality of the latent space"

img_shape = (channels, img_size_time, img_size_freq)


# VAE Network Architecture -------------------------------------
class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


# -----------------------
# Residual Block (Used in Both Encoder & Decoder)
# -----------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, upsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.upsample = upsample

        # Main convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity mapping)
        self.skip = nn.Sequential()
        if in_channels != out_channels or downsample or upsample:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        # Downsample or Upsample
        self.downsample_layer = nn.AvgPool2d(2) if downsample else nn.Identity()
        self.upsample_layer = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            if upsample
            else nn.Identity()
        )

    def forward(self, x):
        x = self.downsample_layer(x)  # Apply downsampling if needed
        x = self.upsample_layer(x)  # Apply upsampling if needed

        identity = self.skip(x)

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity  # Add skip connection
        return F.relu(x)


# -----------------------
# Residual-based Encoder
# -----------------------
class Encoder(nn.Module):
    def __init__(
        self,
        img_channels=1,
        img_height=img_size_time,
        img_width=img_size_freq,
        base_channels=32,
        latent_dim=latent_dim,
    ):
        super(Encoder, self).__init__()

        self.initial_conv = nn.Conv2d(
            img_channels, base_channels, kernel_size=3, stride=1, padding=0
        )

        # ResNet Blocks (Downsampling)
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_channels, base_channels * 2, downsample=True),
            ResidualBlock(base_channels * 2, base_channels * 4, downsample=True),
            ResidualBlock(base_channels * 4, base_channels * 8, downsample=True),
            ResidualBlock(base_channels * 8, base_channels * 8, downsample=True),
        )

        # Compute final feature map size
        self.final_h = img_height // 16
        self.final_w = img_width // 16
        self.fc_mu = nn.Linear(
            base_channels * 8 * self.final_h * self.final_w, latent_dim
        )
        self.fc_logvar = nn.Linear(
            base_channels * 8 * self.final_h * self.final_w, latent_dim
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = x.view(x.shape[0], -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar  # Mean and Log Variance for VAE


# -----------------------
# Residual-based Decoder
# -----------------------
class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim=latent_dim,
        base_channels=16,
        img_channels=1,
        img_height=img_size_time,
        img_width=img_size_freq,
    ):
        super(Decoder, self).__init__()

        # Compute initial feature map size
        self.init_h = img_height // 16
        self.init_w = img_width // 16
        self.fc = nn.Linear(
            latent_dim, base_channels * 8 * (self.init_h + 1) * (self.init_w + 1)
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(base_channels * 8, base_channels * 8, upsample=True),
            ResidualBlock(base_channels * 8, base_channels * 4, upsample=True),
            ResidualBlock(base_channels * 4, base_channels * 2, upsample=True),
            ResidualBlock(base_channels * 2, base_channels, upsample=True),
        )

        self.final_conv1 = nn.Sequential(
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=5,
                stride=1,
                padding=(2,0),
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.final_conv2 = nn.Sequential(
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=(1,0),
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.final_conv3 = nn.Sequential(
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                stride=1,
                padding=(1,0),
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.final_conv4 = nn.Conv2d(
            base_channels * 2,
            img_channels, 
            kernel_size=3, 
            stride=1, 
            padding=0
        )

    def forward(self, z):
        x = self.fc(z).view(z.shape[0], -1, (self.init_h + 1), (self.init_w + 1))
        x = self.res_blocks(x)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        x = self.final_conv3(x)
        x = self.final_conv4(x)
        return torch.sigmoid(x)  # Normalize output to [0,1]



        

# ResNet Architecture ------------------------------------------

class ResNet18classification(nn.Module):
    def __init__(self, weights_path=None):
        super(ResNet18classification, self).__init__()

        # load resnet18
        self.model = models.resnet18(weights=None)

        if weights_path is not None:
            state_dict = torch.load(weights_path, weights_only=True)
            self.model.load_state_dict(state_dict)

        # adapt to grayscale input
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # change classifier head → single logit
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)  # raw logits 