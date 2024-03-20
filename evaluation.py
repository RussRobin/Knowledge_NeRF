"""
evaluation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import os
import imageio
import numpy as np
from PIL import Image

from torchvision.models import inception_v3
from torchvision import transforms
from scipy import linalg

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Mean Square Error
class MSE(object):
    def __call__(self, pred, gt):
        print('pred.shape,gt.shape',pred.shape,gt.shape)
        return torch.mean((pred - gt) ** 2)

# Peak Signal to Noise Ratio
class PSNR(object):
    def __call__(self, pred, gt):
        mse = torch.mean((pred - gt) ** 2)
        return 10 * torch.log10(1 / mse)

# structural similarity index
class SSIM(object):
    '''
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    '''
    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

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

# Learned Perceptual Image Patch Similarity
class LPIPS(object):
    '''
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    '''
    def __init__(self):
        self.model = lpips.LPIPS(net='vgg').cuda()

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        error =  self.model.forward(y_pred, y_true)
        return torch.mean(error)

class FID(object):
    def __init__(self, batch_size=50, dims=2048, cuda=True):
        self.batch_size = batch_size
        self.dims = dims
        self.cuda = cuda
        self.device = torch.device("cuda" if cuda else "cpu")
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()

    def compute_activations(self, images):
        """
        Compute activations of the pool_3 layer for each image.
        """
        with torch.no_grad():
            activations = self.inception_model(images)[0]

        return activations

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """
        Calculate Frechet distance between two multivariate Gaussians.
        """
        eps = 1e-6

        diff = mu1 - mu2
        print(sigma1)
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def calculate_activation_statistics(self, images):
        """
        Calculate the mean and covariance of activations for a set of images.
        """
        act = self.compute_activations(images)
        print(act.shape)
        # mu = np.mean(act, axis=0, dtype=np.float64)  # Explicitly specify dtype
        mu = np.mean(act.cpu().numpy(), axis=0, dtype=np.float64)  # Explicitly specify dtype
        sigma = np.cov(act.cpu().numpy(), rowvar=False)
        return mu, sigma

    def preprocess_images(self, images):
        """
        Preprocess images for the InceptionV3 model.
        """
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        images = 2 * images - 1  # Normalize to [-1, 1]

        return images

    def __call__(self, y_pred, y_true):
        """
        Compute Frechet Inception Distance (FID) between two sets of images.
        Args:
            y_true (torch.Tensor): Real images, 4D tensor [batch_size, channels, img_rows, img_cols]
            y_pred (torch.Tensor): Generated images, 4D tensor [batch_size, channels, img_rows, img_cols]
        Returns:
            fid_score (float): Frechet Inception Distance
        """
        y_true = self.preprocess_images(y_true)
        y_pred = self.preprocess_images(y_pred)

        mu1, sigma1 = self.calculate_activation_statistics(y_true)
        mu2, sigma2 = self.calculate_activation_statistics(y_pred)

        fid_score = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        return fid_score

def read_images_in_dir(imgs_dir,is_gt=False):
    imgs = []
    fnames = os.listdir(imgs_dir)
    image_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    fnames = [fname for fname in fnames if os.path.splitext(fname)[-1].lower() in image_formats]

    fnames.sort()

    for fname in fnames:
        img_path = os.path.join(imgs_dir, fname)
        img = imageio.imread(img_path)

        img = (np.array(img) / 255.).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        img = img[:3, :, :]

        imgs.append(img)
    
    imgs = np.stack(imgs)       
    return imgs

def estim_error(estim, gt):
    errors = dict()
    metric = MSE()
    errors["mse"] = metric(estim, gt).item()
    metric = PSNR()
    errors["psnr"] = metric(estim, gt).item()
    metric = SSIM()
    errors["ssim"] = metric(estim, gt).item()
    metric = LPIPS()
    errors["lpips"] = metric(estim, gt).item()
    return errors

def save_error(errors, name, save_dir):
    save_path = os.path.join(save_dir, name+".txt")
    f = open(save_path,"w")
    f.write( str(errors) )
    f.close()

### specify name and path here
name = '...'
estim_dir = '...'
gt_dir = '...'

savedir = '...'
### 

estim = read_images_in_dir(estim_dir)
gt = read_images_in_dir(gt_dir,is_gt=True)

estim = torch.Tensor(estim).cuda()
gt = torch.Tensor(gt).cuda()

errors = estim_error(estim, gt)
save_error(errors, name, savedir)
print(errors)