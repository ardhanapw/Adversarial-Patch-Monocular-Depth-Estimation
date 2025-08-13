import torch
import random
import numpy as np
import cv2


def resize_img(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


class Compose(object):
    '''Set of tranform random routines that takes list of inputs as arguments,
    in order to have random but coherent transformations.'''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images


class RandomAugumentColor(object):
    """Randomly color augument the given numpy array"""

    def __call__(self, images):
        random_gamma = np.random.uniform(0.8, 1.2)
        gamma_images = [im ** random_gamma for im in images]

        random_brightness = np.random.uniform(0.8, 1.2)
        random_colors = np.random.uniform(0.95, 1.05, [1, 1, 3]) * random_brightness
        output_images = [np.clip(im*random_colors, 0, 255) for im in gamma_images]
        return output_images


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images):
        if random.random() < 0.5:
            output_images = [np.copy(np.fliplr(im)) for im in images]
        else:
            output_images = images

        return output_images


class ResizeImage(object):
    """Randomly zooms images up to 15% and crop them to a particular size"""

    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, images):
        resized_images = [resize_img(im, self.w, self.h) for im in images]
        return resized_images


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to a particular size"""

    def __init__(self, scale_range=(0.85, 1.0)):
        self.min_scale, self.max_scale = scale_range

    def __call__(self, images):
        if self.min_scale < 1.0:
            scale = np.random.uniform(self.min_scale, self.max_scale)
        else:
            scale = 1.0

        in_h, in_w, _ = images[0].shape
        scaled_h, scaled_w = int(in_h * scale), int(in_w * scale)

        offset_y = np.random.randint(in_h - scaled_h + 1)
        offset_x = np.random.randint(in_w - scaled_w + 1)

        cropped_images = [im[offset_y:offset_y + scaled_h, offset_x:offset_x + scaled_w] for im in images]
        output_images = [resize_img(im, in_w, in_h) for im in cropped_images]

        return output_images


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images):
        tensors = []
        for im in images:
            im = np.transpose(im, (2, 0, 1))
            tensors.append(torch.from_numpy(im).float() / 255.)
        return tensors
