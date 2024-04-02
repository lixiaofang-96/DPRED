import torch
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import glob
import random
import cv2


def Tensor(img):  # from numpy to tensor
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(np.ascontiguousarray(img)).float()
    return img


def Array(img):
    img = img[0, 0].detach().numpy()  # from tensor to numpy
    # img = img.transpose(1, 2, 0) # [c,m,n]->[m,n,c]
    return img


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


class MyDataset(torch.utils.data.Dataset):
    """
    Loads and transforms images before feeding it to the first layer of the network.
    Attributes
    ----------
        folder      (str): path to the folder containing the images
        file_names (list): list of strings, list of names of images
        file_list  (list): list of strings, paths to images
        need_names  (str): 'yes' for outputting image names, 'no' else
    """

    def __init__(self, rand_mode, folder='/path/to/folder/'):
        """
        Loads and transforms images before feeding it to the network.
        Parameters
        ----------
        folder     (str): path to the folder containing the images (default '/path/to/folder/')
        need_names (str): 'yes' for outputting image names, 'no' else (default is 'no')
        """
        super(MyDataset, self).__init__()
        self.train_low_folder, self.train_high_folder = folder
        self.train_file_names = glob.glob(self.train_low_folder + '*.*')
        self.train_file_list = [os.path.join(self.train_low_folder, i) for i in self.train_file_names]
        self.num = 0
        self.patch_size = 48
        self.rand_mode = rand_mode

    def __getitem__(self, index):
        """
        Loads and transforms an image.
        Parameters
        ----------
            index (int): index of the image in the list of files, can point to a .mat, .jpg, .png.
                         If the image has just one channel the function will convert it to an RGB format by
                         repeating the channel.
       Returns
       -------
                          (str): optional, image name without the extension
            (torch.FloatTensor): image before transformation, size c*h*w
            (torch.FloatTensor): image after transformation, size c*h*w
        """
        # .jpg or .png file
        low_img = Image.open(self.train_file_list[index])
        # low_img = cv2.imread(self.train_file_list[index])
        # low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2HSV)
        low_img = np.asarray(low_img) / 255.0
        # low_img = (low_img - np.min(low_img)) / (np.max(low_img) - np.min(low_img))
        name = os.path.basename(self.train_file_names[index])[:-4].split('\\')[-1]

        high_path = self.train_high_folder + name + '.png'
        high_img = Image.open(high_path)
        # high_img = cv2.imread(high_path)
        # high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2HSV)
        high_img = np.asarray(high_img) / 255.0
        # high_img = (high_img - np.min(high_img)) / (np.max(high_img) - np.min(high_img))

        h, w, _ = low_img.shape
        x = random.randint(0, h - self.patch_size)
        y = random.randint(0, w - self.patch_size)
        input_low = Tensor(data_augmentation(
            low_img[x: x + self.patch_size, y: y + self.patch_size, :], self.rand_mode))
        input_high = Tensor(data_augmentation(
            high_img[x: x + self.patch_size, y: y + self.patch_size, :], self.rand_mode))

        return name, input_low, input_high

    def __len__(self):
        return len(self.train_file_list)