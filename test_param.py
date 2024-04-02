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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_param(eval_floder):
    root = './Retinex_result/'
    path_save = root + 'test_param/'
    os.makedirs(path_save, exist_ok=True)

    Decom_net = DecomNet_RTV(in_ch=1)
    Decom_net.cuda()
    print('====>>   load Decom_Net\n')

    pre_Decom_checkpoint = './checkpoint/decomnet_V_train_new/checkpoint_99_epoch.pkl'
    checkpoint = torch.load(pre_Decom_checkpoint)
    Decom_net.load_state_dict(checkpoint['model_state_dict'])

    eval_data = glob.glob(eval_floder[0] + '79.*')
    eval_data.sort()
    num = len(eval_data)
    PSNR = []
    SSIM = []
    with torch.no_grad():
        for path_mat in eval_data:
            name = os.path.basename(path_mat)[:-4].split('\\')[-1]
            print('validating image:', name)
            input_eval_low = np.asarray(Image.open(eval_floder[0] + name + '.png'))
            input_eval_high = np.asarray(Image.open(eval_floder[1] + name + '.png'))
            input_low = Tensor(input_eval_low).to(device)
            # [b, c, h, w] = input_low.shape
            # d = 32
            # input_low = F.pad(input_low, (0, d - w % d, 0, d - h % d))
            HSV_low = rgb_to_hsv(input_low)
            input_H_low = HSV_low[:, 0].unsqueeze(1)
            input_S_low = HSV_low[:, 1].unsqueeze(1)
            input_V_low = HSV_low[:, 2].unsqueeze(1)

            low_I_list, low_R_list, alpha, px, py, mu = Decom_net(input_V_low)
            low_I, low_R = low_I_list[-1], low_R_list[-1]

            enhance_I = low_I ** (1 / 2.2)
            enhance_V = enhance_I * low_R

            enhance_img = torch.cat([input_H_low, input_S_low, enhance_V], dim=1)
            # high_I_list, high_R_list, alpha, px, py, mu = Decom_net(input_V_high)
            enhance_img = hsv_to_rgb(enhance_img)

            enhance_high = enhance_img[0].cpu().detach().numpy()
            enhance_high = np.transpose(enhance_high, (1, 2, 0)) * 255.0
            enhance_high = np.clip(enhance_high, 0, 255.0)

            psnr_enhance = psnr(enhance_high, input_eval_high)
            ssim_enhance = ssim(enhance_high, input_eval_high)
            print("eval image: %s, PSNR = %5.4f, SSIM = %5.4f" % (name, psnr_enhance, ssim_enhance))
            PSNR.append(psnr_enhance)
            SSIM.append(ssim_enhance)
            with open('./Retinex_result/test_param/metric.txt', 'a') as f:
                f.write("eval image: %s, PSNR = %5.4f, SSIM = %5.4f" % (name, psnr_enhance, ssim_enhance))
                f.write("\n")

            alpha = alpha[0].cpu().numpy()
            px = px[0].cpu().numpy()
            py = py[0].cpu().numpy()
            plt.figure()

            ax = plt.gca()
            # plt.hist(alpha[0].flatten(), range=[0.0003, 0.0009], bins=120, density=1, rwidth=0.8)
            im_alpha = ax.imshow(alpha[0],cmap=plt.cm.hot_r)
            # plt.xlabel('$\lambda$')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_alpha, cax=cax)
            # plt.xticks([])
            # plt.yticks([])
            ax.axis('off')
            plt.savefig(os.path.join(path_save, '%s_alpha' % name), dpi=1200, bbox_inches='tight', )
            plt.clf()
            plt.figure()
            ax = plt.gca()
            # plt.hist(px[0].flatten(), range=[0.0, 0.8], bins=120, density=1, rwidth=0.8)
            im_px = ax.imshow(px[0], cmap=plt.cm.hot_r)
            # plt.xlabel('$p$')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_px, cax=cax)
            # plt.xticks([])
            # plt.yticks([])
            ax.axis('off')
            plt.savefig(os.path.join(path_save, '%s_px' % name), dpi=1200, bbox_inches='tight')
            plt.clf()
            plt.figure()
            ax = plt.gca()
            # plt.hist(py[0].flatten(), range=[0.0, 0.8], bins=120, density=1, rwidth=0.8)
            im_py = ax.imshow(py[0], cmap=plt.cm.hot_r)
            # plt.xlabel('$q$')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_py, cax=cax)
            # plt.xticks([])
            # plt.yticks([])
            ax.axis('off')
            plt.savefig(os.path.join(path_save, '%s_py' % name), dpi=1200, bbox_inches='tight')
            plt.clf()
            # plt.show()
            # save_alpha(os.path.join(path_save, '%s' % name), alpha)
            # save_px(os.path.join(path_save, '%s' % name), wx)
            # save_py(os.path.join(path_save, '%s' % name), wy)
            # np.savetxt(os.path.join(path_save, '%s_mu.txt' % name), mu[0, :, 0, 0].cpu().detach().numpy(), fmt='%d')

    avg_PSNR = np.mean(np.asarray(PSNR))
    avg_SSIM = np.mean(np.asarray(SSIM))
    print("avg_PSNR = %5.4f, avg_SSIM = %5.4f" % (avg_PSNR, avg_SSIM))
    with open('./Retinex_result/test_param/metric.txt', 'a') as f:
        f.write("avg_PSNR = %5.4f, avg_SSIM = %5.4f" % (avg_PSNR, avg_SSIM))
        f.write("\n")


if __name__ == '__main__':
    eval_folder = ['/home/www/myRetinex/data/LOL/eval15/low/', '/home/www/myRetinex/data/LOL/eval15/high/']
    # eval_folder = ['/media/www/14F492BBF4929F14/水下图像增强/raw-890-s/raw-890/', '/media/www/14F492BBF4929F14'
    #                                                                        '/水下图像增强/reference-890/reference-890/']
    # eval_folder = ['/media/www/EXTERNAL_USB/MIT5K/default/', '/media/www/EXTERNAL_USB/MIT5K/ExpertsC/']
    test_param(eval_folder)
