
##############################################################
#  Seismic Signal Denoising via Self-Supervised Res2-UNet
#  Full Research-Level Implementation
##############################################################

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

##############################################################
# DEVICE
##############################################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

##############################################################
# METRICS
##############################################################

def mse(x, y):
    return ((x - y) ** 2).mean().item()

def psnr(x, y):
    mse_val = ((x - y) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse_val + 1e-8))

##############################################################
# J-INVARIANT MASKING
##############################################################

def j_invariant_mask(x, mask_ratio=0.1):

    mask = torch.rand_like(x) < mask_ratio

    kernel = torch.ones((1,1,3,3), device=x.device)/9.0
    blurred = F.conv2d(x, kernel, padding=1)

    x_masked = x.clone()
    x_masked[mask] = blurred[mask]

    return x_masked, mask

##############################################################
# DATASET
##############################################################

def extract_patches(img, patch=128, stride=32):

    patches = []
    H, W = img.shape

    for i in range(0, H-patch, stride):
        for j in range(0, W-patch, stride):
            patches.append(img[i:i+patch, j:j+patch])

    return np.array(patches)


class SeismicDataset(Dataset):

    def __init__(self, n_samples=200):

        self.data = []

        for _ in range(n_samples):
            clean = np.random.randn(512,512)*0.2

            for k in range(10):
                clean += np.sin(np.linspace(0,20,512))[None,:]

            noise = np.random.normal(0,0.3,(512,512))
            noisy = clean + noise

            patches = extract_patches(noisy)
            self.data.extend(patches)

        self.data = np.array(self.data, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.tensor(x).unsqueeze(0)
        return x

##############################################################
# RES2NET BLOCK
##############################################################

class Res2Block(nn.Module):
    def __init__(self, channels, scale=4):
        super().__init__()

        width = channels // scale
        self.scale = scale

        self.convs = nn.ModuleList([
            nn.Conv2d(width,width,3,padding=1)
            for _ in range(scale-1)
        ])

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):

        splits = torch.chunk(x,self.scale,dim=1)
        outputs=[splits[0]]

        for i in range(1,self.scale):
            out = splits[i] + outputs[i-1]
            out = self.convs[i-1](out)
            outputs.append(out)

        out = torch.cat(outputs,dim=1)
        out = self.bn(out)
        return self.relu(out)

##############################################################
# UNET BUILDING BLOCKS
##############################################################

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch,out_ch)

    def forward(self,x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch,out_ch,2,stride=2)
        self.conv = DoubleConv(in_ch,out_ch)

    def forward(self,x1,x2):

        x1 = self.up(x1)

        diffY = x2.size()[2]-x1.size()[2]
        diffX = x2.size()[3]-x1.size()[3]

        x1 = F.pad(x1,[diffX//2,diffX-diffX//2,
                       diffY//2,diffY-diffY//2])

        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)

##############################################################
# RES2-UNET MODEL
##############################################################

class Res2UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.inc = DoubleConv(1,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)

        self.res2 = Res2Block(512)

        self.up1 = Up(512,256)
        self.up2 = Up(256,128)
        self.up3 = Up(128,64)

        self.outc = nn.Conv2d(64,1,1)

    def forward(self,x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.res2(x4)

        x = self.up1(x5,x3)
        x = self.up2(x,x2)
        x = self.up3(x,x1)

        return self.outc(x)

##############################################################
# SELF-SUPERVISED LOSS
##############################################################

def self_supervised_loss(model,noisy):

    masked_input,mask = j_invariant_mask(noisy)

    output = model(masked_input)

    loss = ((output-noisy)**2)[mask].mean()

    return loss, output

##############################################################
# TRAINING FUNCTION
##############################################################

def train():

    dataset = SeismicDataset()
    loader = DataLoader(dataset,batch_size=8,shuffle=True)

    model = Res2UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    epochs = 20

    for epoch in range(epochs):

        model.train()
        total_loss=0
        total_psnr=0

        for noisy in tqdm(loader):

            noisy=noisy.to(DEVICE)

            optimizer.zero_grad()

            loss,output = self_supervised_loss(model,noisy)

            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            total_psnr+=psnr(output.detach(),noisy).item()

        print(f"Epoch {epoch+1}")
        print("Loss:",total_loss/len(loader))
        print("PSNR:",total_psnr/len(loader))

    torch.save(model.state_dict(),"res2unet_seismic.pth")

    return model

##############################################################
# INFERENCE + VISUALIZATION
##############################################################

def inference(model):

    model.eval()

    sample = torch.randn(1,1,128,128).to(DEVICE)

    with torch.no_grad():
        output = model(sample)

    sample = sample.cpu().squeeze()
    output = output.cpu().squeeze()

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Noisy")
    plt.imshow(sample,cmap="gray")

    plt.subplot(1,2,2)
    plt.title("Denoised")
    plt.imshow(output,cmap="gray")

    plt.show()

##############################################################
# MAIN
##############################################################

if __name__=="__main__":

    model = train()
    inference(model)
