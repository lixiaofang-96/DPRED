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
from network_unet import UNetRes as net
from model import DecomNet_RTV, EnhanceNet_I
import pytorch_ssim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_Denoising_enI(dataloader, eval_floder, numBatch):
    root = './Retinex_result/'
    path_save = root + 'Denoise_enI_train_DnCNN/'
    os.makedirs(path_save, exist_ok=True)
    path_save_eval = root + 'Denoise_enI_eval_DnCNN/'
    os.makedirs(path_save_eval, exist_ok=True)

    Decom_net = DecomNet_RTV(in_ch=1, k1=10)
    Decom_net.cuda()
    print('====>>   load Decom_Net\n')

    pre_Decom_checkpoint = './checkpoint/decomnet_V_train_new/checkpoint_99_epoch.pkl'
    checkpoint = torch.load(pre_Decom_checkpoint)
    Decom_net.load_state_dict(checkpoint['model_state_dict'])

    Enhance_I = EnhanceNet_I(in_ch=2, out_ch=1).cuda()
    pre_EnhanceI_checkpoint = './checkpoint/Enhancenet_I_train_FD/checkpoint_99_epoch.pkl'
    checkpoint = torch.load(pre_EnhanceI_checkpoint)
    Enhance_I.load_state_dict(checkpoint['I_model_state_dict'])

    Enhance_R = EnhanceNet_I(in_ch=2, out_ch=1).cuda()
    pre_EnhanceR_checkpoint = './checkpoint/Enhancenet_VR_train_new/checkpoint_99_epoch.pkl'
    checkpoint = torch.load(pre_EnhanceR_checkpoint)
    Enhance_R.load_state_dict(checkpoint['R_model_state_dict'])

    lr = 1e-5

    n_channels = 3
    noise_level_img = 15  # set AWGN noise level for noisy image
    noise_level_model = noise_level_img  # set noise level for model
    Denoise_enI = net(in_nc=n_channels + 1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                      downsample_mode="strideconv", upsample_mode="convtranspose").cuda()
    Denoise_enI.load_state_dict(torch.load('/home/www/PycharmProjects/myRetinex_new/checkpoint/Denoise_enI_train_new/'
                                           'drunet_color.pth'), strict=True)
    Denoise_enI_op = torch.optim.Adam(Denoise_enI.parameters(), lr=lr)
    scheduler_enI = torch.optim.lr_scheduler.StepLR(Denoise_enI_op, step_size=100, gamma=0.1)  # 设置学习率下降策略

    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    ssim_loss = pytorch_ssim.SSIM()
    start_epoch = 0
    MAX_epoch = 100
    eval_every_epoch = 10
    checkpoint_interval = 10
    log_interval = 20
    train_phase = 'Denoising_enI'
    start_time = time.time()

    checkpoint_dir = './checkpoint/Denoise_enI_train_DnCNN/'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ckpt = False
    if ckpt:
        path_checkpoint = './checkpoint/Denoise_enI_train_DnCNN/checkpoint_9_epoch.pkl'
        checkpoint = torch.load(path_checkpoint)
        Denoise_enI.load_state_dict(checkpoint['enI_model_state_dict'])
        Denoise_enI_op.load_state_dict(checkpoint['enI_optimizer_state_dict'])
        # scheduler_H.load_state_dict(checkpoint['H_lr_schedule'])
        start_epoch = checkpoint['epoch']
        # scheduler_H.last_epoch = start_epoch

    for epoch in range(start_epoch, MAX_epoch):
        lc_time = time.asctime(time.localtime(time.time()))
        Denoise_enI.train()
        for i, batch in enumerate(dataloader):
            input_low, input_high = Variable(batch[1]), Variable(batch[2], requires_grad=False)

            train_low_data = input_low.to(device)
            train_high_data = input_high.to(device)

            HSV_train_low_data = rgb_to_hsv(train_low_data)
            HSV_train_high_data = rgb_to_hsv(train_high_data)

            train_low_V = HSV_train_low_data[:, 2].unsqueeze(1)
            train_high_V = HSV_train_high_data[:, 2].unsqueeze(1)

            low_I_list, low_R_list, alpha, px, py, mu = Decom_net(train_low_V)
            high_I_list, high_R_list, alpha, px, py, mu = Decom_net(train_high_V)

            low_I, low_R, high_I, high_R = low_I_list[-1], low_R_list[-1], high_I_list[-1], high_R_list[-1]

            I_ratio = low_I / (high_I + 0.0001)
            enhance_I = Enhance_I(low_I, I_ratio)
            enhance_R = Enhance_R(low_R, low_I)
            enhance_img = enhance_I * enhance_R

            train_low = torch.cat(
                [HSV_train_low_data[:, 0].unsqueeze(1), HSV_train_low_data[:, 1].unsqueeze(1), enhance_img], dim=1)
            enhance_img = hsv_to_rgb(train_low)

            img_L = enhance_img
            img_L = torch.cat(
                (img_L, torch.FloatTensor([noise_level_model / 255.]).repeat(img_L.shape[0], 1, img_L.shape[2],
                                                                             img_L.shape[3]).to(device)),
                dim=1)

            denoise_enI = Denoise_enI(img_L)

            Denoise_enI_op.zero_grad()
            loss = mse(denoise_enI, train_high_data.detach()) + 1 - ssim_loss(denoise_enI, train_high_data)
            loss.backward()
            Denoise_enI_op.step()

            # for b in range(batch[1].shape[0]):
            #     name = batch[0][b]
            #     save_enhance_images(os.path.join(path_save, '%s_%d_%d' % (name, i + 1, epoch + 1)), denoise_enI)

            if (i + 1) % log_interval == 0:
                print("%s %s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss_I: %.6f" \
                      % (train_phase, lc_time, epoch + 1, i + 1, numBatch, time.time() - start_time,
                         loss.data.cpu().numpy()))

            scheduler_enI.step()  # 更新学习率

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {"epoch": epoch,
                          "enI_model_state_dict": Denoise_enI.state_dict(),
                          "enI_optimizer_state_dict": Denoise_enI_op.state_dict()
                          # 'H_lr_schedule': scheduler_H.state_dict(),
                          }
            path_checkpoint = checkpoint_dir + "/checkpoint_{}_epoch.pkl".format(epoch)
            torch.save(checkpoint, path_checkpoint)

        if (epoch + 1) % eval_every_epoch == 0:
            print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
            eval_data = glob.glob(eval_floder[0] + '*.png')
            num = len(eval_data)
            PSNR = []
            SSIM = []
            with torch.no_grad():
                for path_mat in eval_data:
                    name = os.path.basename(path_mat)[:-4].split('\\')[-1]
                    print('validating image:', name)
                    eval_low = np.asarray(Image.open(eval_floder[0] + name + '.png'))
                    eval_high = np.asarray(Image.open(eval_floder[1] + name + '.png'))

                    input_eval_low = Tensor(eval_low).to(device)
                    input_eval_high = Tensor(eval_high).to(device)

                    input_eval_low = rgb_to_hsv(input_eval_low)
                    input_eval_high = rgb_to_hsv(input_eval_high)

                    eval_low_V = input_eval_low[:, 2].unsqueeze(1)
                    eval_high_V = input_eval_high[:, 2].unsqueeze(1)

                    low_I_list, low_R_list, alpha, px, py, mu = Decom_net(eval_low_V)
                    high_I_list, high_R_list, alpha, px, py, mu = Decom_net(eval_high_V)
                    # low_I_list, low_R_list, alpha, px, py, mu = Decom_net(input_eval_low)
                    # high_I_list, high_R_list, alpha, px, py, mu = Decom_net(input_eval_high)

                    low_I, low_R, high_I, high_R = low_I_list[-1], low_R_list[-1], high_I_list[-1], high_R_list[-1]

                    I_ratio = low_I / (high_I + 0.0001)
                    enhance_I = Enhance_I(low_I, I_ratio)
                    enhance_R = Enhance_R(low_R, low_I)
                    enhance_img = enhance_I * enhance_R

                    # input_eval_low[:, 2] = enhance_img
                    HSV_eval_low = torch.cat(
                        [input_eval_low[:, 0].unsqueeze(1), input_eval_low[:, 1].unsqueeze(1), enhance_img], dim=1)
                    enhance_img = hsv_to_rgb(HSV_eval_low)

                    img_L = enhance_img
                    img_L = torch.cat(
                        (img_L,
                         torch.FloatTensor([noise_level_model / 255.]).repeat(img_L.shape[0], 1, img_L.shape[2],
                                                                              img_L.shape[3]).to(
                             device)),
                        dim=1)

                    denoise_enI = Denoise_enI(img_L)

                    enhance_high = denoise_enI[0].cpu().detach().numpy()
                    enhance_high = np.transpose(enhance_high, (1, 2, 0)) * 255.0
                    enhance_high = np.clip(enhance_high, 0, 255.0)

                    psnr_enhance = psnr(enhance_high, eval_high)
                    ssim_enhance = ssim(enhance_high, eval_high)
                    print("eval image: %s, PSNR = %5.4f, SSIM = %5.4f" % (name, psnr_enhance, ssim_enhance))
                    PSNR.append(psnr_enhance)
                    SSIM.append(ssim_enhance)

                    save_enhance_images(os.path.join(path_save_eval, '%s_%d_%d' % (name, i + 1, epoch + 1)),
                                        denoise_enI)
            avg_PSNR = np.mean(np.asarray(PSNR))
            avg_SSIM = np.mean(np.asarray(SSIM))
            print("avg_PSNR = %5.4f, avg_SSIM = %5.4f" % (avg_PSNR, avg_SSIM))
            ## compute PSNR and SSIM
    print("[*] Finish training for phase %s." % train_phase)


if __name__ == '__main__':
    train_folder = ['/media/www/14F492BBF4929F14/data/LOL/our485/low/', '/media/www/14F492BBF4929F14/data/LOL/our485/high/']
    batch_size = 10
    train_Data = []
    for patch_id in range(batch_size):
        rand_mode = np.random.randint(0, 7)
        train_data = MyDataset(rand_mode, train_folder)
        train_Data.extend(train_data)
    print('[*] Number of training data: %d' % len(train_Data))
    numBatch = len(train_Data) // int(batch_size)

    eval_floder = ['/media/www/14F492BBF4929F14/data/LOL/eval15/low/', '/media/www/14F492BBF4929F14/data/LOL/eval15/high/']
    dataloader = DataLoader(dataset=train_Data, batch_size=batch_size, shuffle=True, num_workers=0,
                            drop_last=True)
    train_Denoising_enI(dataloader, eval_floder, numBatch)
