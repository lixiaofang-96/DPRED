import torch
import numpy as np
from PIL import Image
import os
import cv2, math
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
from torch.autograd import Variable
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def Tensor(img):  # from numpy to tensor
    img = img.transpose(2, 0, 1) / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    return img


def save_images(filepath, result):
    I, R = result[0].cpu().detach().numpy(), result[1].cpu().detach().numpy()

    I = np.transpose(I, (1, 2, 0)) * 255.0
    R = np.transpose(R, (1, 2, 0)) * 255.0

    I = np.clip(I, 0, 255.0)
    R = np.clip(R, 0, 255.0)

    cv2.imwrite(filepath + '_I.png', I)
    cv2.imwrite(filepath + '_R.png', R)

    # I, R, S, T = result[0].cpu().detach().numpy(), result[1].cpu().detach().numpy(), result[2].cpu().detach().numpy(), \
    #              result[3].cpu().detach().numpy()
    #
    # I = np.transpose(I, (1, 2, 0)) * 255.0
    # R = np.transpose(R, (1, 2, 0)) * 255.0
    # S = np.transpose(S, (1, 2, 0)) * 255.0
    # T = np.transpose(T, (1, 2, 0)) * 255.0
    #
    # I = np.clip(I, 0, 255.0)
    # R = np.clip(R, 0, 255.0)
    # S = np.clip(S, 0, 255.0)
    # T = np.clip(T, 0, 255.0)
    #
    # I[:, :, [2, 1, 0]] = I[:, :, [0, 1, 2]]
    # cv2.imwrite(filepath + '_I.png', I)
    # R[:, :, [2, 1, 0]] = R[:, :, [0, 1, 2]]
    # cv2.imwrite(filepath + '_R.png', R)
    # S[:, :, [2, 1, 0]] = S[:, :, [0, 1, 2]]
    # cv2.imwrite(filepath + '_S.png', S)
    # T[:, :, [2, 1, 0]] = T[:, :, [0, 1, 2]]
    # cv2.imwrite(filepath + '_T.png', T)
    # I = Image.fromarray(np.clip(I * 255.0, 0, 255.0).astype('uint8'))
    # I.save(filepath + '_I.png')
    # S = Image.fromarray(np.clip(S * 255.0, 0, 255.0).astype('uint8'))
    # S.save(filepath + '_S.png')
    # T = Image.fromarray(np.clip(T * 255.0, 0, 255.0).astype('uint8'))
    # T.save(filepath + '_T.png')
    # R = Image.fromarray(np.clip(R * 255.0, 0, 255.0).astype('uint8'))
    # R.save(filepath + '_R.png')


def save_eval_images(filepath, result):
    I, R = result[0][0].cpu().detach().numpy(), result[1][0].cpu().detach().numpy()

    I = np.transpose(I, (1, 2, 0)) * 255.0
    R = np.transpose(R, (1, 2, 0)) * 255.0

    I = np.clip(I, 0, 255.0)
    R = np.clip(R, 0, 255.0)

    I[:, :, [2, 1, 0]] = I[:, :, [0, 1, 2]]
    cv2.imwrite(filepath + '_I.png', I)
    R[:, :, [2, 1, 0]] = R[:, :, [0, 1, 2]]
    cv2.imwrite(filepath + '_R.png', R)


def save_eval_images_V(filepath, result):
    I, R = result[0][0].cpu().detach().numpy(), result[1][0].cpu().detach().numpy()

    I = np.transpose(I, (1, 2, 0)) * 255.0
    R = np.transpose(R, (1, 2, 0)) * 255.0

    I = np.clip(I, 0, 255.0)
    R = np.clip(R, 0, 255.0)

    cv2.imwrite(filepath + '_I.png', I)
    cv2.imwrite(filepath + '_R.png', R)


def save_enhance_images(filepath, result):
    img = result[0].cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = np.clip(img, 0, 255.0)
    img[:, :, [2, 1, 0]] = img[:, :, [0, 1, 2]]
    cv2.imwrite(filepath + '_enhance.png', img)


def save_enhance_V(filepath, result):
    img = result[0].cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = np.clip(img, 0, 255.0)
    cv2.imwrite(filepath + '_V_enhance.png', img)


def save_enhance_R(filepath, result):
    [b, c, h, w] = result.shape
    img = result[0].cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = np.clip(img, 0, 255.0)
    if c == 3:
        img[:, :, [2, 1, 0]] = img[:, :, [0, 1, 2]]
    cv2.imwrite(filepath + '_R.png', img)


def save_alpha(filepath, result):
    img = result[0].cpu().detach().numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = np.clip(img, 0, 255.0)
    cv2.imwrite(filepath + '_lambda.png', img)


def save_px(filepath, result):
    img = result[0].cpu().detach().numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = np.clip(img, 0, 255.0)
    cv2.imwrite(filepath + '_p.png', img)


def save_py(filepath, result):
    img = result[0].cpu().detach().numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = np.clip(img, 0, 255.0)
    cv2.imwrite(filepath + '_q.png', img)


def save_enhance_I(filepath, result):
    [b, c, h, w] = result.shape
    img = result[0].cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = np.clip(img, 0, 255.0)
    if c == 3:
        img[:, :, [2, 1, 0]] = img[:, :, [0, 1, 2]]
    cv2.imwrite(filepath + '_I.png', img)


def save_denoise_H(filepath, result):
    img = result[0].cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = np.clip(img, 0, 255.0)
    cv2.imwrite(filepath + '_H.png', img)


def save_denoise_S(filepath, result):
    img = result[0].cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = np.clip(img, 0, 255.0)
    cv2.imwrite(filepath + '_S.png', img)


def save_enhance_S(filepath, result):
    img = result[0].cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = np.clip(img, 0, 255.0)
    # img[:, :, [2, 1, 0]] = img[:, :, [0, 1, 2]]
    cv2.imwrite(filepath + '_S.png', img)


def save_enhance_T(filepath, result):
    img = result[0].cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = np.clip(img, 0, 255.0)
    # img[:, :, [2, 1, 0]] = img[:, :, [0, 1, 2]]
    cv2.imwrite(filepath + '_T.png', img)


def grad(x):
    x_diff = x[:, :, :, 1:] - x[:, :, :, :-1]
    x_e = (x[:, :, :, 0] - x[:, :, :, -1]).unsqueeze(3)
    grd_x = torch.cat((x_diff, x_e), 3).to(device)
    y_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
    y_e = (x[:, :, 0, :] - x[:, :, -1, :]).unsqueeze(2)
    grd_y = torch.cat((y_diff, y_e), 2).to(device)
    return grd_x, grd_y


def fftn(t, batch, channel, row, col, dim):
    y = torch.fft.fft(t, col, dim=dim)
    y = y.expand(batch, channel, col, row)
    return y


def fftnt(t, batch, channel, row, col, dim):
    y = torch.fft.fft(t, col, dim=dim)
    y = y.expand(batch, channel, row, col)
    return y


def Dive(x, y):
    x_diff = x[:, :, :, :-1] - x[:, :, :, 1:]
    x_e = (x[:, :, :, -1] - x[:, :, :, 0]).unsqueeze(3)
    x_diff = torch.cat((x_e, x_diff), 3)
    y_diff = y[:, :, :-1, :] - y[:, :, 1:, :]
    y_e = (y[:, :, -1, :] - y[:, :, 0, :]).unsqueeze(2)
    y_diff = torch.cat((y_e, y_diff), 2)
    return y_diff + x_diff


def ForwardDiff(x):
    x_diff = x[:, :, :, 1:] - x[:, :, :, :-1]
    x_e = (x[:, :, :, 0] - x[:, :, :, -1]).unsqueeze(3)
    x_diff = torch.cat((x_diff, x_e), 3)
    y_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
    y_e = (x[:, :, 0, :] - x[:, :, -1, :]).unsqueeze(2)
    y_diff = torch.cat((y_diff, y_e), 2)
    return x_diff, y_diff


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1: size // 2 + 1, -size // 2 + 1: size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    g = g / g.sum()
    g = torch.from_numpy(g).float()
    return g


def lpfilter(f, sigma):
    ksize = round(5 * sigma) | 1
    g = fspecial_gauss(ksize, sigma)
    g = g.unsqueeze(0).unsqueeze(0)
    g = g.repeat(1, f.shape[1], 1, 1).to(device)
    weight = nn.Parameter(data=g, requires_grad=False)
    ret = F.conv2d(f, weight, stride=1, padding=ksize // 2, dilation=1, groups=1)
    return ret


def computeweights(f, sigma):
    gf = lpfilter(f, sigma)
    gfx, gfy = grad(gf)
    wtbx = torch.max(torch.abs(gfx), torch.tensor(0.001)) ** -1
    wtby = torch.max(torch.abs(gfy), torch.tensor(0.001)) ** -1
    wtbx[:, :, :, -1] = 0
    wtby[:, :, -1, :] = 0
    return wtbx, wtby


def shrink(x, r, m):
    z = torch.sign(x) * torch.max(torch.abs(x) - r, m)
    return z


def hsv_to_rgb(hsv):
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    # 对出界值的处理
    h = h % 1
    s = torch.clamp(s, 0, 1)
    v = torch.clamp(v, 0, 1)

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))

    hi0 = hi == 0
    hi1 = hi == 1
    hi2 = hi == 2
    hi3 = hi == 3
    hi4 = hi == 4
    hi5 = hi == 5

    r[hi0] = v[hi0]
    g[hi0] = t[hi0]
    b[hi0] = p[hi0]

    r[hi1] = q[hi1]
    g[hi1] = v[hi1]
    b[hi1] = p[hi1]

    r[hi2] = p[hi2]
    g[hi2] = v[hi2]
    b[hi2] = t[hi2]

    r[hi3] = p[hi3]
    g[hi3] = q[hi3]
    b[hi3] = v[hi3]

    r[hi4] = t[hi4]
    g[hi4] = p[hi4]
    b[hi4] = v[hi4]

    r[hi5] = v[hi5]
    g[hi5] = p[hi5]
    b[hi5] = q[hi5]

    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    rgb = torch.cat([r, g, b], dim=1)

    return rgb


def rgb_to_hsv(img):
    eps = 1e-8
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
    saturation[img.max(1)[0] == 0] = 0

    value = img.max(1)[0]

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value], dim=1)
    return hsv


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1, img2):
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0) / 255.0
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0) / 255.0
    img1 = Variable(img1, requires_grad=False)  # torch.Size([256, 256, 3])
    img2 = Variable(img2, requires_grad=False)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    return ssim_value


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image
