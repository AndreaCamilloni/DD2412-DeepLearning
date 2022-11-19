# Data Loader 

import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,  datasets
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.image_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.image_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


class ImageFolderWithIndices(datasets.ImageFolder):
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithIndices, self).__getitem__(index)
        # make a new tuple that includes original and the index
        tuple_with_path = (original_tuple + (index,))
        return tuple_with_path

# Augmentation returning 2 augmented images
class Augmentation:
    def __init__(self, transform):
        self.transform = transform
    

    def __call__(self, x):
        out = list()
        out.append(self.transform(x))
        out.append(self.transform(x))
        #x1 = self.transform(x)
        #x2 = self.transform(x)
        return out

# Random Transformation
def get_transform():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform