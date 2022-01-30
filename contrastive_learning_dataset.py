from torch.utils.data.dataset import Dataset
# from torchvision.transforms import transforms
# from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
# from data_aug.view_generator import ContrastiveLearningViewGenerator
# from exceptions.exceptions import InvalidDatasetSelection
from torchvision.datasets.folder import default_loader
import csv
import os
import pandas as pd
from skimage import io
from augmentations import get_aug


class ContrastiveLearningDataset(Dataset):
    def __init__(self, txt_file, type, name, img_size, train, train_classifier=None, loader = default_loader):
        fh = open(txt_file, 'r')
        imgs = []
        self.type = type
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()  # 分割成文件名和标签
            if self.type == "train":
                imgs.append((words[0], words[1]))
            else:
                imgs.append(words[0])
        self.imgs = imgs
        self.name = name
        self.img_size = img_size
        self.transform = get_aug(self.name, image_size=self.img_size, train=train, train_classifier=train_classifier)
        # self.transform = self.get_simclr_pipeline_transform(84)


    # @staticmethod
    # def get_simclr_pipeline_transform(size, s=1):
    #     """Return a set of data augmentation transformations as described in the SimCLR paper."""
    #     imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
    #     color_jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
    #     data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           transforms.RandomApply([color_jitter], p=0.8),
    #                                           transforms.RandomGrayscale(p=0.2),
    #                                           transforms.RandomApply([transforms.GaussianBlur(kernel_size=int(size//20*2+1), sigma=(0.1, 2.0))], p=0.5),
    #                                           transforms.ToTensor(),
    #                                           transforms.Normalize(*imagenet_mean_std)])
        
            
    #     return data_transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.type == "train":
            fn, label = self.imgs[index]
        else:
            fn = self.imgs[index]
        # original_img = io.imread(fn)
        original_img = default_loader(fn)
        if self.transform is not None:
            img = self.transform(original_img) 
        if self.type == "train":
            return img, label
        else:
            return img