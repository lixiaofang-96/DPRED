import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse, datetime, time
import os
from glob import glob
from MyDataset import *
from myutils import *
from loss import *
from model import EnhanceNet_I, DecomNet_RTV
from network_unet import UNetRes as net
import torch.nn.functional as F

# device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_HSV(eval_floder):
    root = '/media/www/EXTERNAL_USB/myRetinex/Retinex_result/'
    path_save = root + 'test_NPE_D1/'
    os.makedirs(path_save, exist_ok=True)
    path_save_eval = root + 'Enhance_result/'
    os.makedirs(path_save_eval, exist_ok=True)

    Decom_net = DecomNet_RTV(in_ch=1,k1=10).to(device)
    print('====>>   load Decom_Net\n')

    pre_Decom_checkpoint = './checkpoint/decomnet_V_train_new/checkpoint_99_epoch.pkl'
    checkpoint = torch.load(pre_Decom_checkpoint)
    Decom_net.load_state_dict(checkpoint['model_state_dict'])

    Enhance_I = EnhanceNet_I(in_ch=2, out_ch=1).to(device)
    pre_EnhanceI_checkpoint = './checkpoint/Enhancenet_I_train_FD/checkpoint_99_epoch.pkl'
    checkpoint = torch.load(pre_EnhanceI_checkpoint)
    Enhance_I.load_state_dict(checkpoint['I_model_state_dict'])

    Enhance_R = EnhanceNet_I(in_ch=2, out_ch=1).to(device)
    pre_EnhanceR_checkpoint = './checkpoint/Enhancenet_VR_train_new/checkpoint_99_epoch.pkl'
    checkpoint = torch.load(pre_EnhanceR_checkpoint)
    Enhance_R.load_state_dict(checkpoint['R_model_state_dict'])

    n_channels = 3
    noise_level_img = 5  # set AWGN noise level for noisy image
    noise_level_model = noise_level_img  # set noise level for model
    Denoise_enI = net(in_nc=n_channels + 1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                      downsample_mode="strideconv", upsample_mode="convtranspose").to(device)
    # Denoise_enI.load_state_dict(torch.load('./checkpoint/Denoise_enI_train_new/drunet_color.pth'), strict=True)
    pre_Denoise_checkpoint = './checkpoint/Denoise_enI_train_new/checkpoint_99_epoch.pkl'
    checkpoint = torch.load(pre_Denoise_checkpoint)
    Denoise_enI.load_state_dict(checkpoint['enI_model_state_dict'])


    eval_data = glob.glob(eval_floder[0] + '*.*')
    eval_data.sort()
    with torch.no_grad():
        for path_mat in eval_data:
            name = os.path.basename(path_mat)[:-4].split('\\')[-1]
            print('validating image:', name)
            input_low = Image.open(path_mat)
            # input_high = Image.open(eval_floder[0] + name + '.png')
            # input_high = input_high.resize((input_low.size[0], input_low.size[1]))

            input_low = Tensor(np.asarray(input_low)).to(device)
            # input_high = Tensor(np.asarray(input_high)).to(device)
            [b, c, h, w] = input_low.shape
            d = 8
            input_low = F.pad(input_low, (0, d - w % d, 0, d - h % d))
            # input_high = F.pad(input_high, (0, d - w % d, 0, d - h % d))

            HSV_low = rgb_to_hsv(input_low)
            input_H_low = HSV_low[:, 0].unsqueeze(1)
            input_S_low = HSV_low[:, 1].unsqueeze(1)
            input_V_low = HSV_low[:, 2].unsqueeze(1)

            # HSV_high = rgb_to_hsv(input_high)
            # input_H_high = HSV_high[:, 0].unsqueeze(1)
            # input_S_high = HSV_high[:, 1].unsqueeze(1)
            # input_V_high = HSV_high[:, 2].unsqueeze(1)

            low_I_list, low_R_list, alpha, px, py, mu = Decom_net(input_V_low)
            # high_I_list, high_R_list, alpha, px, py, mu = Decom_net(input_V_high)

            low_I, low_R = low_I_list[-1], low_R_list[-1]
            # high_I, high_R = high_I_list[-1], high_R_list[-1]

            save_enhance_I(os.path.join(path_save, 'low_%s' % name), low_I[:, :, :h, :w])
            save_enhance_R(os.path.join(path_save, 'low_%s' % name), low_R[:, :, :h, :w])

            I_ratio = low_I / ((low_I ** (1 / 3)) + 0.0001)
            # I_ratio = low_I / (high_I + 0.0001)
            enhance_I = Enhance_I(low_I, I_ratio)
            enhance_R = Enhance_R(low_R, low_I)
            enhance_img = enhance_I * enhance_R

            enhance_img = torch.cat([input_H_low, input_S_low, enhance_img], dim=1)
            enhance_img = hsv_to_rgb(enhance_img)

            img_L = enhance_img
            img_L = torch.cat(
                (img_L,
                 torch.FloatTensor([noise_level_model / 255.]).repeat(img_L.shape[0], 1, img_L.shape[2],
                                                                      img_L.shape[3]).to(
                     device)),
                dim=1)

            enhance_img = Denoise_enI(img_L)
            # print(enhance_img.shape)
            enhance_img = enhance_img[:, :, :h, :w]
            enhance_I = enhance_I[:, :, :h, :w]
            enhance_R = enhance_R[:, :, :h, :w]

            save_enhance_images(os.path.join(path_save, '%s' % name), enhance_img)
            save_enhance_I(os.path.join(path_save, '%s' % name), low_I[:, :, :h, :w])
            save_enhance_R(os.path.join(path_save, '%s' % name), low_R[:, :, :h, :w])


if __name__ == '__main__':
    # eval_floder = ['/home/www/myRetinex/data/LOL/BrighteningTrain/low/','/home/www/myRetinex/data/LOL/BrighteningTrain/high/']
    eval_floder = ['/media/www/14F492BBF4929F14/data/NPE_1/']
    # eval_floder = ['/home/www/myRetinex/data/LIME/', '/media/www/14F492BBF4929F14/fsdownload/KinD-plus_results/LIME/']
    # eval_floder = ['/media/www/EXTERNAL_USB/LLIE代码/VV/']
    # eval_floder = ['/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/数据集/ExDark/People/']
    test_HSV(eval_floder)
