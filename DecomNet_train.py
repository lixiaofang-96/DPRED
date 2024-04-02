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
import argparse
from Enhance_I_train import *
from Enhance_R_train import *
from Denoise_enI_train import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_DecomNet(dataloader, eval_floder, args):
    ### 训练参数
    lambda_rc = args.lambda_rc
    lambda_imc = args.lambda_imc
    print('show {}  {}'.format(lambda_rc, lambda_imc))

    ### 设置保存路径
    root = './Retinex_result/'
    path_save = root + 'Decom_V_train_RGB'  # + str(lambda_rc) + str(lambda_imc) + '/'
    print(path_save)
    os.makedirs(path_save, exist_ok=True)
    path_save_eval = root + 'Decom_V_eval_RGB'  # + str(lambda_rc) + str(lambda_imc) + '/'
    os.makedirs(path_save_eval, exist_ok=True)
    ### 创建网络模型
    print('====>>   Build Net\n')
    Decom_net = DecomNet_RTV(in_ch=3, k1=10)
    Decom_net.apply(weights_init_kaiming)  ### 模型初始化
    Decom_net.to(device)  ### 把模型放到GPU上

    ### 设置学习率，优化器，学习率调整策略
    learning_rate = 1e-4
    Decom_op = torch.optim.Adam(Decom_net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(Decom_op, step_size=10, gamma=0.1)  # 设置学习率下降策略
    start_epoch = 0
    start_time = time.time()

    ### 保存模型训练节点 保存训练好的模型参数
    checkpoint_dir = './checkpoint/decomnet_V_train_RGB'  # + str(lambda_rc) + str(lambda_imc) + '/'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ckpt = False
    if ckpt:
        path_checkpoint = './checkpoint/decomnet_V_train_RGB' + '/checkpoint_9_epoch.pkl'
        # + str(lambda_rc) + str(lambda_imc)
        checkpoint = torch.load(path_checkpoint)
        Decom_net.load_state_dict(checkpoint['model_state_dict'])
        Decom_op.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['lr_schedule'])
        scheduler.last_epoch = start_epoch

    ### 设置损失
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

            # HSV_low = rgb_to_hsv(train_low_data)
            # HSV_high = rgb_to_hsv(train_high_data)
            # input_H_low = HSV_low[:, 0].unsqueeze(1)
            # input_S_low = HSV_low[:, 1].unsqueeze(1)
            # input_V_low = HSV_low[:, 2].unsqueeze(1)
            #
            # input_H_high = HSV_high[:, 0].unsqueeze(1)
            # input_S_high = HSV_high[:, 1].unsqueeze(1)
            # input_V_high = HSV_high[:, 2].unsqueeze(1)

            low_I_list, low_R_list, alpha, px, py, mu = Decom_net(train_low_data)
            high_I_list, high_R_list, alpha, px, py, mu = Decom_net(train_high_data)

            # low_I_list, low_R_list, alpha, px, py, mu = Decom_net(train_low_data)
            # high_I_list, high_R_list, alpha, px, py, mu = Decom_net(train_high_data)

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
            # recon_loss = l1(train_low_data, low_I * low_R) + l1(train_high_data, high_I * high_R)
            loss = recon_loss + lambda_rc * equal_R_loss + lambda_imc * mutual_I_loss  # + mutual_R1_loss + mutual_R2_loss
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
                    # HSV_low = rgb_to_hsv(input_eval_low)
                    # HSV_high = rgb_to_hsv(input_eval_high)
                    # input_H_low = HSV_low[:, 0].unsqueeze(1)
                    # input_S_low = HSV_low[:, 1].unsqueeze(1)
                    # input_V_low = HSV_low[:, 2].unsqueeze(1)
                    # input_H_high = HSV_high[:, 0].unsqueeze(1)
                    # input_S_high = HSV_high[:, 1].unsqueeze(1)
                    # input_V_high = HSV_high[:, 2].unsqueeze(1)
                    # low_I_list, low_R_list, alpha, px, py, mu = Decom_net(input_V_low)
                    # high_I_list, high_R_list, alpha, px, py, mu = Decom_net(input_V_high)

                    low_I_list, low_R_list, alpha, px, py, mu = Decom_net(input_eval_low)
                    high_I_list, high_R_list, alpha, px, py, mu = Decom_net(input_eval_high)

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

                    # save_eval_images(os.path.join(path_save_eval, 'low/', '%s_%d_%d' % (name, i + 1, epoch + 1)),
                    #                    low_result)
                    # save_eval_images(os.path.join(path_save_eval, 'high/', '%s_%d_%d' % (name, i + 1, epoch + 1)),
                    #                    high_result)

                    # save_alpha(os.path.join(path_save_eval, 'low/', '%s_%d_%d' % (name, i + 1, epoch + 1)), p)
                    # save_alpha(os.path.join(path_save_eval, 'high/', '%s_%d_%d' % (name, i + 1, epoch + 1)), p)
                    # save_beta(os.path.join(path_save_eval, 'low/', '%s_%d_%d' % (name, i + 1, epoch + 1)), q)
                    # save_beta(os.path.join(path_save_eval, 'high/', '%s_%d_%d' % (name, i + 1, epoch + 1)), q)
                    # np.savetxt(os.path.join(path_save_eval, 'high/', '%s_%d_%d_mu.txt' % (name, i + 1, epoch + 1)),
                    #            mu[0, :, 0, 0].cpu().detach().numpy(), fmt='%d')

    print("[*] Finish training for phase %s." % train_phase)


def set_random_seed(seed=42):
    torch.manual_seed(seed)  # torch的cpu随机性
    torch.cuda.manual_seed_all(seed)  # torch的gpu随机性
    torch.backends.cudnn.benchmark = False  # 保证gpu每次都选择相同的算法，但是不保证该算法是deterministic的。
    torch.backends.cudnn.deterministic = True  # 紧接着上面，保证算法是deterministic的。
    np.random.seed(seed)  # np的随机性。
    random.seed(seed)  # python的随机性。
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置python哈希种子，有人不知道这个是干啥的，
    # python里面有很多使用哈希算法完成的操作，例如对于一个数字的列表，使用set()来去重。
    # 大家应该经历过，得到的结果中，顺序可能不一样，例如(1,2,3)(3,2,1)。
    # 有时候需要在终端就把这个固定执行，到脚本实行有可能会迟。


if __name__ == '__main__':
    set_random_seed(42)

    # 1. 定义命令行解析器对象
    parser = argparse.ArgumentParser(description='Demo of argparse')

    # 2. 添加命令行参数
    parser.add_argument('--lambda_rc', type=float, default=0.1)
    parser.add_argument('--lambda_imc', type=float, default=0.1)
    # 3. 从命令行中结构化解析参数
    args = parser.parse_args()
    print(args)

    ### 生成训练数据
    train_folder = ['/media/www/14F492BBF4929F14/data/LOL/our485/low/', '/media/www/14F492BBF4929F14/data/LOL/our485/high/']
    batch_size = 10
    train_Data = []
    for patch_id in range(batch_size):
        rand_mode = np.random.randint(0, 7)
        train_data = MyDataset(rand_mode, train_folder)
        train_Data.extend(train_data)
    print('[*] Number of training data: %d' % len(train_Data))
    numBatch = len(train_Data) // int(batch_size)

    ### 导入验证数据
    eval_floder = ['/media/www/14F492BBF4929F14/data/LOL/eval15/low/', '/media/www/14F492BBF4929F14/data/LOL/eval15/high/']

    ### 生成训练数据
    dataloader = DataLoader(dataset=train_Data, batch_size=batch_size, shuffle=True, num_workers=0,
                            drop_last=True)
    ### 训练网络
    train_DecomNet(dataloader, eval_floder, args)
    # train_EnhanceNet_I(dataloader, eval_floder, numBatch)
    # train_EnhanceNet_R(dataloader, eval_floder, numBatch)
    # train_Denoising_enI(dataloader, eval_floder, numBatch)


