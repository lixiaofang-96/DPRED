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
from model import DecomNet_RTV

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_DecomNet(dataloader, eval_floder):
    root = '/media/www/EXTERNAL_USB/myRetinex/Retinex_result/'
    path_save = root + 'Decom_V_train_k15/'
    os.makedirs(path_save, exist_ok=True)
    path_save_eval = root + 'Decom_V_eval_k15/'
    os.makedirs(path_save_eval, exist_ok=True)

    print('====>>   Build Net\n')
    Decom_net = DecomNet_RTV(in_ch=1, k1=15)
    Decom_net.apply(weights_init_kaiming)
    Decom_net.to(device)

    learning_rate = 1e-4
    Decom_op = torch.optim.Adam(Decom_net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(Decom_op, step_size=10, gamma=0.1)  # 设置学习率下降策略
    start_epoch = 0
    start_time = time.time()

    checkpoint_dir = './checkpoint/decomnet_V_train_k15/'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ckpt = False
    if ckpt:
        path_checkpoint = './checkpoint/decomnet_V_train_k15/checkpoint_89_epoch.pkl'
        checkpoint = torch.load(path_checkpoint)
        Decom_net.load_state_dict(checkpoint['model_state_dict'])
        Decom_op.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['lr_schedule'])
        scheduler.last_epoch = start_epoch

    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    train_phase = 'decomposition'

    MAX_epoch = 100
    eval_every_epoch = 10
    checkpoint_interval = 10
    log_interval = 20

    for epoch in range(start_epoch, MAX_epoch):
        lc_time = time.asctime(time.localtime(time.time()))
        Decom_net.train()
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

            Decom_op.zero_grad()

            low_I, low_R, high_I, high_R = low_I_list[-1], low_R_list[-1], high_I_list[-1], high_R_list[-1]

            # print(low_I, low_R)
            # for b in range(batch[1].shape[0]):
            #     name = batch[0][b]
            #     low_result = [low_I[b], low_R[b]]
            #     high_result = [high_I[b], high_R[b]]
            # save_images(os.path.join(path_save, 'low/', '%s_%d_%d' % (name, i + 1, epoch + 1)), low_result)
            # save_images(os.path.join(path_save, 'high/', '%s_%d_%d' % (name, i + 1, epoch + 1)), high_result)

            equal_R_loss = l1(low_R, high_R)
            mutual_I_loss = mutual_loss(low_I, high_I)
            mutual_R1_loss = grad_loss(low_R, input_high)
            mutual_R2_loss = grad_loss(high_R, input_high)
            recon_loss = l1(train_low_data, low_I * low_R) + l1(train_high_data, high_I * high_R)
                         # + l1(train_low_data, low_I * high_R) + l1(train_high_data, high_I * low_R)
            loss = recon_loss + 0.1 * equal_R_loss + 0.1 * mutual_I_loss #+ mutual_R1_loss + mutual_R2_loss
            # print(recon_loss, mutual_I_loss)
            # loss = 0.1 * equal_R_loss
            loss.backward()
            # for name, parms in Decom_net.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
            #           ' -->grad_value:', parms.grad)
            # nn.utils.clip_grad_norm(Decom_net.parameters(), 1, norm_type=2)
            Decom_op.step()
            if (i + 1) % log_interval == 0:
                print("%s %s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, lc_time, epoch + 1, i + 1, numBatch, time.time() - start_time,
                         loss.data.cpu().detach().numpy()))

        scheduler.step()  # 更新学习率

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": Decom_net.state_dict(),
                          "optimizer_state_dict": Decom_op.state_dict(),
                          "epoch": epoch,
                          'lr_schedule': scheduler.state_dict()}
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

                    # low_I_list, low_R_list, px, py, qx, qy, mu = Decom_net(input_eval_low)
                    # high_I_list, high_R_list, px, py, qx, qy, mu = Decom_net(input_eval_high)

                    '''
                    low_I_list, low_R_list, alpha, px, py, mu = Decom_net(input_eval_low)
                    high_I_list, high_R_list, alpha, px, py, mu = Decom_net(input_eval_high)
                    '''

                    low_I, low_R, high_I, high_R = low_I_list[-1], low_R_list[-1], high_I_list[-1], high_R_list[-1]

                    low_result = [low_I, low_R]
                    high_result = [high_I, high_R]
                    save_eval_images_V(os.path.join(path_save_eval, 'low/', '%s_%d_%d' % (name, i + 1, epoch + 1)),
                                       low_result)
                    save_eval_images_V(os.path.join(path_save_eval, 'high/', '%s_%d_%d' % (name, i + 1, epoch + 1)),
                                       high_result)

                    # save_alpha(os.path.join(path_save_eval, 'low/', '%s_%d_%d' % (name, i + 1, epoch + 1)), p)
                    # save_alpha(os.path.join(path_save_eval, 'high/', '%s_%d_%d' % (name, i + 1, epoch + 1)), p)
                    # save_beta(os.path.join(path_save_eval, 'low/', '%s_%d_%d' % (name, i + 1, epoch + 1)), q)
                    # save_beta(os.path.join(path_save_eval, 'high/', '%s_%d_%d' % (name, i + 1, epoch + 1)), q)
                    # np.savetxt(os.path.join(path_save_eval, 'high/', '%s_%d_%d_mu.txt' % (name, i + 1, epoch + 1)),
                    #            mu[0, :, 0, 0].cpu().detach().numpy(), fmt='%d')

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
    train_DecomNet(dataloader, eval_floder)
