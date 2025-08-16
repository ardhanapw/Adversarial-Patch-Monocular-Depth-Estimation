import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import random
from PIL import Image
from torchvision import transforms as T

def load_patch_from_img(path):
    #RGB patch from image
    img = Image.open(path).convert('RGB')
    patch = T.ToTensor()(img)
    return patch

class BaseDataset(Dataset):
    def __init__(self, image_list_file):
        self.image_file_list = self.load_file_list(image_list_file)

    def load_file_list(self, filenames_file):
        image_file_list = []
        with open(filenames_file) as f:
            lines = f.readlines()
            for line in lines:
                image_file_list.append(line.strip().split())
        return image_file_list

    def load_image(self, path):
        img = cv2.imread(path)
        return img

    def resize_img(self, img, width, height):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


class LoadFromImageFile(BaseDataset):
    def __init__(self, data_path, filenames_file, seed=None, transform=None, extension=None):
        super(LoadFromImageFile, self).__init__(filenames_file)
        np.random.seed(seed)
        random.seed(seed)
        self.root = data_path
        self.transform = transform
        self.extension = extension

        print('=> Load {} images from the paths listed in {}'.format(len(self.image_file_list), self.root + "/" + filenames_file))

    def __getitem__(self, idx):
        left_fn = self.image_file_list[idx][0]
        if self.extension:
            left_fn = os.path.splitext(left_fn)[0]
            image_path = os.path.join(self.root, left_fn) + self.extension
        else:
            image_path = os.path.join(self.root, left_fn)
        image = self.load_image(image_path)
        image = np.expand_dims(image, axis=0)
        if self.transform is not None:
            image = self.transform(image)
        #print("Image shape")
        #print(image.shape)
        sample = {"left": image[0]}
        return sample

    def __len__(self):
        return len(self.image_file_list)