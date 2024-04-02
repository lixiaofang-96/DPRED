import lpips
import glob
import torch
import os
from PIL import Image
import numpy as np
from myutils import *

if __name__ == '__main__':

    eval_floder = ['/media/www/14F492BBF4929F14/data/MIT/high/',
                   '/media/www/14F492BBF4929F14/lixiaofang/study/code/Retinex/CUE-master/results/MIT/']

    eval_data = glob.glob(eval_floder[0] + '*.*')
    eval_data.sort()
    num = len(eval_data)
    PSNR = []
    SSIM = []
    with torch.no_grad():
        for path_mat in eval_data:
            name, format = os.path.basename(path_mat).split('.')
            print('validating image:', name)
            gt_path = np.asarray(Image.open(path_mat))
            # eIm_path = np.asarray(Image.open(eval_floder[1] + name + '.' + format))
            eIm_path = np.asarray(Image.open(eval_floder[1] + name + '.png'))

            psnr_enhance = psnr(eIm_path, gt_path)
            ssim_enhance = ssim(eIm_path, gt_path)
            print("eval image: %s, PSNR = %5.4f, SSIM = %5.4f" % (name, psnr_enhance, ssim_enhance))
            PSNR.append(psnr_enhance)
            SSIM.append(ssim_enhance)
            with open(eval_floder[1] + 'metric.txt', 'a') as f:
                f.write("eval image: %s, PSNR = %5.4f, SSIM = %5.4f" % (name, psnr_enhance, ssim_enhance))
                f.write("\n")

        avg_PSNR = np.mean(np.asarray(PSNR))
        avg_SSIM = np.mean(np.asarray(SSIM))
        print("avg_PSNR = %5.4f, avg_SSIM = %5.4f" % (avg_PSNR, avg_SSIM))
        with open(eval_floder[1] + 'metric.txt', 'a') as f:
            f.write("avg_PSNR = %5.4f, avg_SSIM = %5.4f" % (avg_PSNR, avg_SSIM))
            f.write("\n")
