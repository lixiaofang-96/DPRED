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
from model import DecomNet_RTV, EnhanceNet_I

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_EnhanceNet_I(dataloader, eval_floder, numBatch):
    root = './Retinex_result/'
    path_save = root + 'Enhance_train_I_NR/'
    os.makedirs(path_save, exist_ok=True)
    path_save_eval = root + 'Enhance_eval_I_NR/'
    os.makedirs(path_save_eval, exist_ok=True)

    Decom_net = DecomNet_RTV(in_ch=1, k1=10).to(device)
    # Decom_net.cuda()
    print('====>>   load Decom_Net\n')

    pre_Decom_checkpoint = './checkpoint/decomnet_V_train_new/checkpoint_99_epoch.pkl'
    checkpoint = torch.load(pre_Decom_checkpoint)
    Decom_net.load_state_dict(checkpoint['model_state_dict'])

    I_lr = 1e-4

    Enhance_I = EnhanceNet_I(in_ch=2, out_ch=1).to(device)
    Enhance_I_op = torch.optim.Adam(Enhance_I.parameters(), lr=I_lr)
    scheduler_I = torch.optim.lr_scheduler.StepLR(Enhance_I_op, step_size=10, gamma=0.1)  # 设置学习率下降策略

    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    start_epoch = 0
    MAX_epoch = 100
    eval_every_epoch = 1
    checkpoint_interval = 10
    log_interval = 20
    train_phase = 'enhancement_I'
    start_time = time.time()

    checkpoint_dir = './checkpoint/Enhancenet_I_train_NR/'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ckpt = False
    if ckpt:
        path_checkpoint = './checkpoint/Enhancenet_I_train_NR/checkpoint_9_epoch.pkl'
        checkpoint = torch.load(path_checkpoint)
        Enhance_I.load_state_dict(checkpoint['I_model_state_dict'])
        Enhance_I_op.load_state_dict(checkpoint['I_optimizer_state_dict'])
        scheduler_I.load_state_dict(checkpoint['I_lr_schedule'])
        start_epoch = checkpoint['epoch']
        scheduler_I.last_epoch = start_epoch

    for epoch in range(start_epoch, MAX_epoch):
        lc_time = time.asctime(time.localtime(time.time()))
        Decom_net.eval()
        Enhance_I.train()
        for i, batch in enumerate(dataloader):
            input_low, input_high = Variable(batch[1]), Variable(batch[2], requires_grad=False)

            train_low_data = input_low.to(device)
            train_high_data = input_high.to(device)

            HSV_low = rgb_to_hsv(train_low_data)
            HSV_high = rgb_to_hsv(train_high_data)
            input_H_low = HSV_low[:, 0].unsqueeze(1)
            input_S_low = HSV_low[:, 1].unsqueeze(1)
            input_V_low = HSV_low[:, 2].unsqueeze(1)

            input_H_high = HSV_high[:, 0].unsqueeze(1)
            input_S_high = HSV_high[:, 1].unsqueeze(1)
            input_V_high = HSV_high[:, 2].unsqueeze(1)

            low_I_list, low_R_list, alpha, px, py, mu = Decom_net(input_V_low)
            high_I_list, high_R_list, alpha, px, py, mu = Decom_net(input_V_high)

            low_I, low_R, high_I, high_R = low_I_list[-1], low_R_list[-1], high_I_list[-1], high_R_list[-1]

            # for b in range(batch[1].shape[0]):
            #     name = batch[0][b]
            #     low_result = [low_I[b], low_R[b]]
            #     high_result = [high_I[b], high_R[b]]
            #     save_images(os.path.join(path_save, 'low/', '%s_%d_%d' % (name, i + 1, epoch + 1)), low_result)
            #     save_images(os.path.join(path_save, 'high/', '%s_%d_%d' % (name, i + 1, epoch + 1)), high_result)

            I_ratio = low_I / (high_I + 0.0001)
            enhance_I = Enhance_I(low_I, I_ratio)
            Enhance_I_op.zero_grad()
            loss_I = l1(enhance_I, high_I.detach()) + grad_loss(enhance_I, high_I.detach()) \
                     + l1(input_V_high, enhance_I * high_R)
            loss_I.backward()
            Enhance_I_op.step()

            # for b in range(batch[1].shape[0]):
            #     name = batch[0][b]
            #     save_enhance_I(os.path.join(path_save, '%s_%d_%d' % (name, i + 1, epoch + 1)), enhance_I)

            if (i + 1) % log_interval == 0:
                print("%s %s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss_I: %.6f" \
                      % (train_phase, lc_time, epoch + 1, i + 1, numBatch, time.time() - start_time,
                         loss_I.data.cpu().numpy()))

        # scheduler_I.step()  # 更新学习率

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {"epoch": epoch,
                          "I_model_state_dict": Enhance_I.state_dict(),
                          "I_optimizer_state_dict": Enhance_I_op.state_dict(),
                          'I_lr_schedule': scheduler_I.state_dict(),
                          }
            path_checkpoint = checkpoint_dir + "/checkpoint_{}_epoch.pkl".format(epoch)
            torch.save(checkpoint, path_checkpoint)

        if (epoch + 1) % eval_every_epoch == 0:
            print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
            eval_data = glob.glob(eval_floder[0] + '*.png')
            with torch.no_grad():
                for path_mat in eval_data:
                    name = os.path.basename(path_mat)[:-4].split('\\')[-1]
                    print('validating image:', name)
                    eval_low = np.asarray(Image.open(eval_floder[0] + name + '.png'))
                    eval_high = np.asarray(Image.open(eval_floder[1] + name + '.png'))

                    input_eval_low = Tensor(eval_low).to(device)
                    input_eval_high = Tensor(eval_high).to(device)
                    HSV_low = rgb_to_hsv(input_eval_low)
                    HSV_high = rgb_to_hsv(input_eval_high)
                    input_H_low = HSV_low[:, 0].unsqueeze(1)
                    input_S_low = HSV_low[:, 1].unsqueeze(1)
                    input_V_low = HSV_low[:, 2].unsqueeze(1)
                    input_H_high = HSV_high[:, 0].unsqueeze(1)
                    input_S_high = HSV_high[:, 1].unsqueeze(1)
                    input_V_high = HSV_high[:, 2].unsqueeze(1)
                    low_I_list, low_R_list, alpha, px, py, mu = Decom_net(input_V_low)
                    high_I_list, high_R_list, alpha, px, py, mu = Decom_net(input_V_high)
                    low_I, low_R, high_I, high_R = low_I_list[-1], low_R_list[-1], high_I_list[-1], high_R_list[-1]
                    
                    I_ratio = low_I / (high_I + 0.0001)

                    enhance_I = Enhance_I(low_I, I_ratio)
                    save_enhance_I(os.path.join(path_save_eval, '%s_%d_%d' % (name, i + 1, epoch + 1)),
                                   enhance_I)

                    ## compute PSNR and SSIM

    print("[*] Finish training for phase %s." % train_phase)


if __name__ == '__main__':
    train_folder = ['/home/www/myRetinex/data/LOL/our485/low/', '/home/www/myRetinex/data/LOL/our485/high/']
    batch_size = 10
    train_Data = []
    # for rand_mode in range(1):
    #     train_data = MyDataset(rand_mode, train_folder)
    #     train_Data.extend(train_data)
    for patch_id in range(batch_size):
        rand_mode = np.random.randint(0, 7)
        # print(rand_mode)
        train_data = MyDataset(rand_mode, train_folder)
        train_Data.extend(train_data)
    print('[*] Number of training data: %d' % len(train_Data))
    numBatch = len(train_Data) // int(batch_size)

    eval_floder = ['/home/www/myRetinex/data/LOL/eval15/low/', '/home/www/myRetinex/data/LOL/eval15/high/']

    dataloader = DataLoader(dataset=train_Data, batch_size=batch_size, shuffle=True, num_workers=0,
                            drop_last=True)
    # train_DecomNet(dataloader, eval_floder)
    train_EnhanceNet_I(dataloader, eval_floder)
