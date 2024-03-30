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
from utils import reproduce, calculate_sliding_std, shuffle_input, get_shuffling_mask, generate_random_permutation



paths = os.listdir('data/FMDD/raw/')
for path in paths:
    model = UNet_n2n_un(1, 1).cuda()
    model.train()      
    apply_local_shuffling=False 
    with open('configs/fmdd.yaml', 'r') as f:
        config = yaml.safe_load(f)

    upsampler = nn.Upsample(scale_factor=config['std_kernel_size'], mode='nearest')    
    noisy_orig_np =  np.float32(Image.open('data/FMDD/raw/' + path))
    clean_orig_np = np.float32(Image.open('data/FMDD/gt/' + path))

    # Estimate the std of the noise using high and low masking ratios
    estimated_std = []   
    for mask_ratio in [0.8, 0.2]:
        model = UNet_n2n_un(1, 1).cuda()
        model.train()

        img_H_torch = torch.from_numpy(clean_orig_np).unsqueeze(0).unsqueeze(0).cuda()
        img_L_torch = torch.from_numpy(noisy_orig_np / 255.).unsqueeze(0).unsqueeze(0).cuda()

        criteron = nn.L1Loss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_iterations'])
        for i in range(config['num_iterations']):

            with torch.no_grad():
                mask = torch.rand(1, config['input_channels'], config['img_size'], config['img_size'])  # uniformly distributed between 0 and 1
                mask = (mask < (1. - mask_ratio)).float().cuda()

            output = model(mask*img_L_torch)
            loss = criteron((1- mask)*output, (1- mask)*img_L_torch) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        with torch.no_grad():
            avg =0.
            for _ in range(config['num_predictions']):
                mask = torch.rand(1, config['input_channels'], config['img_size'], config['img_size']) 
                mask = (mask < (1. - mask_ratio)).float().cuda()

                output = model(mask*img_L_torch)
                to_img = output.detach().cpu().squeeze().numpy()
                avg += to_img
                
        #print(mask_ratio, np.std( avg/float(config['num_predictions']) * 255. -noisy_orig_np))
        estimated_std.append(np.std( avg/float(config['num_predictions']) * 255. -noisy_orig_np))


    if abs(estimated_std[0] - estimated_std[1]) > config['epsilon_high']:
        apply_local_shuffling=True
        mask_ratio = config['mask_high']
    elif abs(estimated_std[0] - estimated_std[1]) < config['epsilon_low']:
        mask_ratio = config['mask_low']
    else:
        mask_ratio = config['mask_medium']

    model = UNet_n2n_un(1, 1).cuda()
    model.train()

  
    noisy_original_torch = torch.from_numpy(noisy_orig_np / 255.).unsqueeze(0).unsqueeze(0).cuda()
    noisy_shuffled_torch = noisy_original_torch.detach().clone() # at the first iterations, no shuffling is applied

    criteron = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_iterations'])

    for iter in range(config['num_iterations']):
        with torch.no_grad():
            mask = torch.rand(1, config['input_channels'], config['img_size'], config['img_size'])  # uniformly distributed between 0 and 1
            mask = (mask <  (1. - mask_ratio)).float().cuda()

        output = model(mask*noisy_original_torch)
        loss = criteron((1- mask)*output, (1- mask)*noisy_shuffled_torch) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if iter == config['shuffling_iteration']:
            if apply_local_shuffling == True: 
                avg =0.
                for _ in range(config['num_predictions']):
                    with torch.no_grad():
                        mask = torch.rand(1, config['input_channels'], config['img_size'], config['img_size'])  # uniformly distributed between 0 and 1
                        mask = (mask < (1. - mask_ratio)).float().cuda()
                        output = model(mask*noisy_original_torch)
                        to_img = output.detach().cpu().squeeze().numpy()
                    avg += to_img
                std_map_torch = calculate_sliding_std(torch.from_numpy(avg/float(config['num_predictions']) * 255.).unsqueeze(0).unsqueeze(0).cuda(), upsampler, config['std_kernel_size'], config['std_kernel_size'])
                shuffling_mask = get_shuffling_mask(std_map_torch,  config['masking_threshold'])
                permutation_indices, _ = generate_random_permutation(config['img_size'], config['input_channels'], config['shuffling_tile_size'])
                noisy_shuffled_np= shuffle_input(noisy_orig_np / 255.,  permutation_indices, mask=shuffling_mask, c=config['input_channels'], size=config['img_size'], k=config['shuffling_tile_size'])
                noisy_shuffled_torch = torch.from_numpy(noisy_shuffled_np).unsqueeze(0).cuda()
                #std_map = (std_map_torch.cpu().squeeze()).numpy()

    avg =0.
    for _ in range(config['num_predictions']):
        with torch.no_grad():
            mask = torch.rand(1, config['input_channels'], config['img_size'], config['img_size'])  # uniformly distributed between 0 and 1
            mask = (mask < (1. - mask_ratio)).float().cuda()
            output = model(mask*noisy_original_torch)
            to_img = output.detach().cpu().squeeze().numpy()
        avg += to_img  
    psnr = peak_signal_noise_ratio(np.clip((avg/float(config['num_predictions']))*255., 0., 255.), np.array(clean_orig_np), data_range=255)
    ssim1 = ssim(np.clip((avg/float(config['num_predictions']))*255., 0., 255.), np.array(clean_orig_np), data_range=255)
    print(psnr, ssim1)   
