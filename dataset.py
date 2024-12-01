import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class VideoSRDataset(Dataset):
    def __init__(self, img_files, transform):
        self.img_files = img_files
        self.transform = transform
    
    def __getitem__(self, index):
        input_img = Image.open(self.img_files[index][0]).convert("RGB")
        input_l = Image.open(self.img_files[index][1]).convert("RGB")
        input_r = Image.open(self.img_files[index][2]).convert("RGB")
        gt_img = Image.open(self.img_files[index][3]).convert("RGB")
        name = self.img_files[index][3].split("/")[-1]

        input_img = self.transform(input_img)
        input_l = self.transform(input_l)
        input_r = self.transform(input_r)
        gt_img = self.transform(gt_img)

        return input_img, input_l, input_r, gt_img, name
    
    def __len__(self):
        return len(self.img_files)


