# 3d input (2 channels - 1 real and 1 imag) as input
import random
import numpy as np
import os
import torch

def undersample(img, fold):
    c, h, w = img.shape
    # Find the number of zero padded columns
    # Since symmetric, search only for the left part
    real_part = img[0, :, :]
    imag_part = img[1, :, :]

    zeroCol = []
    for i in range(int(w/2)):
        one_col = real_part[:,i]
        if all(one_col == 0):
            zeroCol.append(i)

    if len(zeroCol) != 0:
        left_i = max(zeroCol) + 1
        right_i = w - left_i

        # Original width before zero padding
        origin_w = right_i - left_i
    
        # number of zero columns
        num_zerocol = w - origin_w
    else:
        origin_w = w

    # number of preserved center column
    if fold == 4:
        center_num = int(origin_w * 0.08)
        preserve_num = int(origin_w * 0.25)
    elif fold == 8:
        center_num = int(origin_w * 0.04)
        preserve_num = int(origin_w * 0.125)
    left = int(w / 2 - center_num / 2)
    right = int(w / 2 + center_num / 2)
    

    # number of columns to make zero
    rest = origin_w - (preserve_num - center_num)
    total_list = list(range(left_i, left)) + list(range(right, right_i))
    rand_sample = random.sample(total_list, rest)
    for j in rand_sample:
        real_part[:, j] = 0
        imag_part[:, j] = 0

    img[0, :, :] = real_part
    img[1, :, :] = imag_part

    return img

def to_original(img):
    # probably 2, 720, 720
    c, h, w = img.shape
    real_part = img[0, :, :]
    imag_part = img[1, :, :]

    zeroCol = []
    for i in range(int(w/2)):
        one_col = real_part[:,i]
        if all(one_col == 0):
            zeroCol.append(i)

    left_i = max(zeroCol) + 1
    right_i = w - left_i
    
    zeroCol = []
    for j in range(int(h/2)):
        one_col = real_part[j,:]
        if all(one_col == 0):
            zeroCol.append(j)
    
    up_i = max(zeroCol) + 1
    down_i = h - up_i
    
    original_img = img[:,up_i:down_i,left_i:right_i]
    return original_img

# center cropping for 2D image

def centeredCrop(img, new_height, new_width):

   width =  np.size(img,1)
   height =  np.size(img,0)

   left = np.int(np.ceil((width - new_width)/2.))
   top = np.int(np.ceil((height - new_height)/2.))
   right = np.int(np.floor((width + new_width)/2.))
   bottom = np.int(np.floor((height + new_height)/2.))
   cImg = img[top:bottom, left:right]
   return cImg

# kspace input (2, 720, 720)
def kspace_recon(sample_k, fold):
    origin_k = to_original(sample_k)
    c,h,w = origin_k.shape
    downi = int((720-h)/2 + h)
    upi = int((720-h)/2)
    righti = int((720-w)/2 + w)
    lefti = int((720-w)/2)
    
    # Ground Truth
    real_gt = origin_k[0,:,:]
    imag_gt = origin_k[1,:,:]
    gt_sum = real_gt + imag_gt * (1j)
    gt = np.fft.fftshift(np.fft.ifft2(gt_sum))
    gt_c = centeredCrop(gt, 320,320)
    
    if fold == 4:
        sample_k = undersample(sample_k, fold)
    elif fold == 8:
        sample_k = undersample(sample_k, fold)
    
    or_sample_k = sample_k[:,upi:downi,lefti:righti]
    # Recon before model
    real_origin = or_sample_k[0,:,:]
    imag_origin = or_sample_k[1,:,:]
    kspace = real_origin + imag_origin * (1j)
    recon = np.fft.fftshift(np.fft.ifft2(kspace))
    recon_c = centeredCrop(recon,320,320)
    
    # Recon after model
    sample_k = torch.from_numpy(sample_k)
    sample_k = sample_k.to(device=device, dtype=torch.float)
    out_k = model(sample_k.view(-1,2,720,720))
    out_np_k = out_k.view(2,720,720).cpu().detach().numpy()
    
    real_out = out_np_k[0,:,:]
    imag_out = out_np_k[1,:,:]
    model_kspace = real_out + imag_out * (1j)
    model_recon = np.fft.fftshift(np.fft.ifft2(model_kspace))
    model_recon_c = centeredCrop(recon,320,320)
    
    
    
    return recon_c, model_recon_c, gt_c

    