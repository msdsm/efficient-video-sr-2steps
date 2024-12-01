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
import os

from ssim import L1_DSSIM_Loss

os.makedirs("./model", exist_ok=True)

config_ini = configparser.ConfigParser()
config_ini.read('config.ini', encoding='utf-8')

EPOCHS = config_ini['TRAIN'].getint('Epochs')
BATCH_SIZE = config_ini['TRAIN'].getint('BatchSize')
ALPHA= config_ini['TRAIN'].getint('Alpha')
ROOT = config_ini['TRAIN']['Root']
CSV_PATH = config_ini['TRAIN']['CsvPath']
RAW_IMAGES_PATH64 = config_ini['TRAIN']['RawImagesPath64']
RAW_IMAGES_PATH128 = config_ini['TRAIN']['RawImagesPath128']
HAT_PATH = config_ini['TRAIN']['HatPath']

alpha = ALPHA / 10
batch_size = BATCH_SIZE
epochs = EPOCHS
suffix = f"{alpha}"
root_path = ROOT

print("=" * 10 + " config " + "=" * 10)
print("alpha : ", alpha)
print("batch_size : ", batch_size)
print("epochs : ", epochs)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=int, default=10)

args = parser.parse_args()

csv_path = root_path + CSV_PATH
raw_images_64_path = root_path + RAW_IMAGES_PATH64
raw_images_128_path = root_path + RAW_IMAGES_PATH128
hat_path = root_path + HAT_PATH
img_files = fetch_imgpath(csv_path, raw_images_64_path, raw_images_128_path, hat_path)

train_img_files = img_files[:int(len(img_files)*0.8)]
val_img_files = img_files[int(len(img_files)*0.8):int(len(img_files)*0.9)]
test_img_files = img_files[int(len(img_files)*0.9):]

print(len(img_files))
print(len(train_img_files), len(val_img_files), len(test_img_files))
print(len(train_img_files)+len(val_img_files)+len(test_img_files))

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = VideoSRDataset(img_files=train_img_files, transform=transform)
val_dataset = VideoSRDataset(img_files=val_img_files, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
print(len(train_dataloader), len(val_dataloader))

model = Unet()

if torch.cuda.is_available():
    device = torch.device("cuda")
    model = nn.DataParallel(model)
else:
    device = torch.device("cpu")

criterion = L1_DSSIM_Loss(alpha = alpha)
optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)

min_val_loss = 10**9
for i in range(epochs):
    print("epoch {}:".format(str(i+1)))
    model.to(device)
    model.train()
    train_loss = 0
    with tqdm(train_dataloader, total=len(train_dataloader)) as pbar:
        for input_img, input_l, input_r, gt_img in pbar:
            input_img = input_img.to(device)
            input_l = input_l.to(device)
            input_r = input_r.to(device)
            gt_img = gt_img.to(device)
            optimizer_model.zero_grad()
            output_img = model(input_img, input_l, input_r)
            loss = criterion(output_img, gt_img)
            loss.backward()
            optimizer_model.step()
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        print("train_loss: {}".format(str(train_loss)))

        model.eval()
        val_loss = 0   

        with torch.no_grad():
            for input_img, input_l, input_r, gt_img in val_dataloader:
                input_img = input_img.to(device)
                input_l = input_l.to(device)
                input_r = input_r.to(device)
                gt_img = gt_img.to(device)
                output_img = model(input_img, input_l, input_r)
                loss = criterion(output_img, gt_img)
                val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
        print("validation loss: {}".format(str(val_loss)))

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            model.to("cpu")
            torch.save(model.module.state_dict(), "./model/" + suffix + ".pth")
            print("save models...")


print("=" * 10 + " config " + "=" * 10)
print("alpha : ", alpha)
print("batch_size : ", batch_size)
print("epochs : ", epochs)