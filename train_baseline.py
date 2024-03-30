import torch
from PIL import Image
import numpy as np
from torch import nn
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import yaml
import random
import os
from model import UNet_n2n_un 
from utils import reproduce

mask_ratio= 0.5

paths = os.listdir('data/FMDD/raw/')
for path in paths:
    model = UNet_n2n_un(1, 1).cuda()
    model.train()      
    with open('configs/fmdd.yaml', 'r') as f:
        config = yaml.safe_load(f)

    noisy_orig_np =  np.float32(Image.open('data/FMDD/raw/' + path))
    clean_orig_np = np.float32(Image.open('data/FMDD/gt/' + path))
    noisy_torch = torch.from_numpy(noisy_orig_np / 255.).unsqueeze(0).unsqueeze(0).cuda()


    criteron = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_iterations'])

    for i in range(config['num_iterations']):

        with torch.no_grad():
            mask = torch.rand(1, config['input_channels'], config['img_size'], config['img_size'])  # uniformly distributed between 0 and 1
            mask = (mask < (1. - mask_ratio)).float().cuda()

        output = model(mask*noisy_torch)
        loss = criteron((1- mask)*output, (1- mask)*noisy_torch) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    with torch.no_grad():
        avg =0.
        for _ in range(config['num_predictions']):
            mask = torch.rand(1, config['input_channels'], config['img_size'], config['img_size']) 
            mask = (mask < (1. - mask_ratio)).float().cuda()

            output = model(mask*noisy_torch)
            to_img = output.detach().cpu().squeeze().numpy()
            avg += to_img         


    psnr = peak_signal_noise_ratio(np.clip((avg/float(config['num_predictions']))*255., 0., 255.), np.array(clean_orig_np), data_range=255)
    ssim1 = ssim(np.clip((avg/float(config['num_predictions']))*255., 0., 255.), np.array(clean_orig_np), data_range=255)
    print(psnr, ssim1)
   
