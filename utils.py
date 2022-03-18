import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from math import exp
import torch
import torch.nn.functional as F


def save_image_grid(images, path=None):

    # create grid from list
    img_grid = make_grid(images, nrow=len(images) if len(images) <= 10 else 10)

    # convert grid to grayscale numpy with hot cmap
    cmap_hot = plt.cm.get_cmap('hot')
    image = img_grid[0, :, :].squeeze(0).numpy()
    image = cmap_hot(image)

    # convert numpy array to Pillow image
    image = Image.fromarray(np.uint8(image * 255))

    # save image
    if path:
        image.save(path)
    return image


def calculate_elapsed_time(start, end):
    """
    Calculates hours, minutes and seconds difference between two timestamps.
    Returns a string in the form: 00:00:00
    """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


def cross_corr(img1, img2):
    """
    Cross-correlation between two images (batch size is expected to be 1!)
    """
    res = F.conv2d(img1, img2) / (torch.norm(img1 * img2, 1))
    return res.item()


# Calculate the one-dimensional Gaussian distribution vector
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# Create a Gaussian kernel, obtained by matrix multiplication of two one-dimensional Gaussian distribution vectors
# You can set the channel parameter to expand to 3 channels
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# Calculate SSIM
# Use the formula of SSIM directly, but when calculating the average value, instead of directly calculating the pixel
# average value, normalized Gaussian kernel convolution is used instead.
# The formula Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y] is used when calculating variance and covariance .
# As mentioned earlier, the above expectation operation is replaced by Gaussian kernel convolution.
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret
