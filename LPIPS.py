import lpips
import glob
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class util_of_lpips():
    def __init__(self, net='vgg', use_gpu=True):
        '''
        Parameters
        ----------
        net: str
            抽取特征的网络，['alex', 'vgg']
        use_gpu: bool
            是否使用GPU，默认不使用
        Returns
        -------
        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img1_path, img2_path):
        '''
        Parameters
        ----------
        img1_path : str
            图像1的路径.
        img2_path : str
            图像2的路径.
        Returns
        -------
        dist01 : torch.Tensor
            学习的感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS).

        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        # Load images
        img0 = lpips.load_image(img1_path)
        img1 = lpips.load_image(img2_path)
        [h, w, c] = img1.shape
        img0 = img0[:h, :w, :]
        img0 = lpips.im2tensor(img0)  # RGB image from [-1,1]
        img1 = lpips.im2tensor(img1)


        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        dist01 = self.loss_fn.forward(img0, img1)
        return dist01


if __name__ == '__main__':
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #               '/media/www/14F492BBF4929F14/Retinex_result/test_LOL_DE/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #               '/media/www/14F492BBF4929F14/Retinex_result/test_LOL_HSV3/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/Retinex_result/test_LOL_HSV6/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/Retinex_result_RGB/test_LOL1/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/RetinexNet-master/test_results/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/KinD-master/results/LOLdataset_eval15/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/KinD_plus-master/test_results/LOLdataset/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/RUAS/result/LOL/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/RUAS-main/result/LOL/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/Zero-DCE-master/Zero-DCE_code/result/LOL/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/Zero-DCE_extension-main/Zero-DCE++/result/LOL/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/Retinex_result/ASWITCH/LOL/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/URetinex-Net-main/result/LOL/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/EnlightenGAN-master/result/LOL/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/home/www/myRetinex/data/LOL/eval15/low/']

    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/Retinex_result_MIT5K/test_MIT2/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/Retinex_result_MIT5K/test_MITD2/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/Retinex_result_MIT5K/test_MIT_ND/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/Retinex_result_MIT5K/test_MIT_D/']

    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/Retinex_result_MIT5K_RGB/test_MITND/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/RetinexNet-master/test_MIT_results/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/KinD-master/results/MIT/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/KinD_plus-master/test_results/MIT/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/RUAS/result/MIT_UPE/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/RUAS-main/result/MIT/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/Zero-DCE-master/Zero-DCE_code/result/MIT/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/Zero-DCE_extension-main/Zero-DCE++/result/MIT/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/Retinex_result/ASWITCH/MIT/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/URetinex-Net-main/result/MIT/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/EnlightenGAN-master/result/MIT/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/EXTERNAL_USB/MIT5K/test/low/']
    # eval_floder = ['/media/www/EXTERNAL_USB/MIT5K/test/high/',
    #                '/media/www/14F492BBF4929F14/Retinex_result_RGB/test_LOL1/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/Retinex_result_MIT5K/test_LOL/']
    # eval_floder = ['/home/www/myRetinex/data/LOL/eval15/high/',
    #                '/home/www/fsdownload/KinD-plus_results/LOL1/']
    # eval_floder = ['/home/www/myRetinex/data/MIT/high/',
    #                '/home/www/fsdownload/RUAS-plus_result/MIT_lol/']
    eval_floder = ['/media/www/14F492BBF4929F14/data/MIT/high/',
                   '/media/www/14F492BBF4929F14/TENR/Result/test/train_gd_num2_iter10_mit_0.85/']

    # eval_floder = ['/media/www/14F492BBF4929F14/data/LOL/eval15/high/',
    #                '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/CUE-master/results/LOL/']
    Lpips = util_of_lpips()
    eval_data = glob.glob(eval_floder[0] + '*.*')
    eval_data.sort()
    num = len(eval_data)
    LPIPS = []
    with torch.no_grad():
        for path_mat in eval_data:
            name, format = os.path.basename(path_mat).split('.')
            print('validating image:', name)
            # gt_path = np.asarray(Image.open(path_mat))
            # eIm_path = np.asarray(Image.open(eval_floder[1] + name + '.' + format))
            gt_path = os.path.join(path_mat)
            # eIm_path = os.path.join(eval_floder[1] + name + '.' + format)
            eIm_path = os.path.join(eval_floder[1] + name + '.png')
            LPIPS_enhance = Lpips.calc_lpips(gt_path, eIm_path)
            print("eval image: %s, LPIPS = %5.4f" % (name, LPIPS_enhance.cpu().numpy()))
            LPIPS.append(LPIPS_enhance.cpu().numpy())
            # with open('/home/www/matlab_code/Retinex/metrics-MIT/our_wD.txt', 'a') as f:
            with open(eval_floder[1] + 'metric.txt', 'a') as f:
            # with open('/media/www/14F492BBF4929F14/Retinex_result_MIT5K/test_LOL/metric.txt', 'a') as f:
                f.write("eval image: %s, LPIPS = %5.4f" % (name, LPIPS_enhance.cpu().numpy()))
                f.write("\n")
    avg_LPIPS = np.mean(np.asarray(LPIPS))
    print("avg_LPIPS = %5.4f" % avg_LPIPS)
    # with open('/home/www/matlab_code/Retinex/metrics-MIT/our_wD.txt', 'a') as f:
    with open(eval_floder[1] + 'metric.txt', 'a') as f:
    # with open('/media/www/14F492BBF4929F14/Retinex_result_MIT5K/test_LOL//metric.txt', 'a') as f:
        f.write("avg_LPIPS = %5.4f" % avg_LPIPS)
        f.write("\n")
