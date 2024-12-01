import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from tqdm import tqdm
from dataset import VideoSRDataset
from fetch_data import fetch_imgpath
from unet import Unet, UnetEncoder
import argparse
import configparser
import cv2


config_ini = configparser.ConfigParser()
config_ini.read('config.ini', encoding='utf-8')

LOAD_PATH = config_ini['TEST']['LoadPath']
ROOT = config_ini['TEST']['Root']
SAVE_PATH = config_ini['TEST']['SavePath']
CSV_PATH = config_ini['TRAIN']['CsvPath']
RAW_IMAGES_PATH64 = config_ini['TRAIN']['RawImagesPath64']
RAW_IMAGES_PATH128 = config_ini['TRAIN']['RawImagesPath128']
HAT_PATH = config_ini['TRAIN']['HatPath']

root_path = ROOT
load_path = LOAD_PATH
save_path = SAVE_PATH

csv_path = root_path + CSV_PATH
raw_images_64_path = root_path + RAW_IMAGES_PATH64
raw_images_128_path = root_path + RAW_IMAGES_PATH128
hat_path = root_path + HAT_PATH
img_files = fetch_imgpath(csv_path, raw_images_64_path, raw_images_128_path, hat_path)
 
test_img_files = img_files[int(len(img_files)*0.9):]
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_dataset = VideoSRDataset(img_files=test_img_files, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# modelの定義
model = Unet()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

model.load_state_dict(torch.load("./model/" + load_path))

model.eval()
psnr_sum = 0.0
cnt = 0
i = 0
with torch.no_grad():
    with tqdm(test_dataloader, total=len(test_dataloader)) as pbar:
        for input_img, input_l, input_r, gt_img, name in pbar:
            input_img = input_img.to(device)
            input_l = input_l.to(device)
            input_r = input_r.to(device)
            gt_img = gt_img.to(device)
            output_img = model(input_img, input_l, input_r)
            output_img = output_img.squeeze(0)
            torchvision.utils.save_image(output_img, save_path + name[0])
                                                                            
            output_img = output_img.to("cpu").detach().numpy().copy().transpose(1, 2, 0).astype(np.float32)
            output_img = np.clip(output_img*255.0, a_min=0, a_max=255).astype(np.uint8)
            gt_img = gt_img.squeeze(0)
            gt_img = gt_img.to("cpu").detach().numpy().copy().transpose(1, 2, 0).astype(np.float32)
            gt_img = np.clip(gt_img*255.0, a_min=0, a_max=255).astype(np.uint8)
            psnr_sum += cv2.PSNR(output_img, gt_img)
            cnt += 1
            i += 1

psnr = psnr_sum / cnt
print("PSNR:", psnr)
print("LOAD_PATH:", load_path)