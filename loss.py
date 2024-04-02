import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def grad(u):
    grd_x = torch.zeros(u.shape).to(device)
    grd_y = torch.zeros(u.shape).to(device)
    grd_x[:, :, :, 0:-1] = u[:, :, :, 0:-1] - u[:, :, :, 1:]
    grd_y[:, :, 0:-1, :] = u[:, :, 0:-1, :] - u[:, :, 1:, :]
    grd_x = torch.abs(grd_x)
    # grd_x = (grd_x - torch.min(grd_x)) / (torch.max(grd_x) - torch.min(grd_x))
    grd_y = torch.abs(grd_y)
    # grd_y = (grd_y - torch.min(grd_y)) / (torch.max(grd_y) - torch.min(grd_y))
    return grd_x, grd_y


def mutual_loss(input_I_low, input_I_high):
    low_grad_x, low_grad_y = grad(input_I_low)
    high_grad_x, high_grad_y = grad(input_I_high)
    x_loss = (low_grad_x + high_grad_x) * torch.exp(-10 * (low_grad_x + high_grad_x))
    y_loss = (low_grad_y + high_grad_y) * torch.exp(-10 * (low_grad_y + high_grad_y))
    mutual_loss = torch.mean(x_loss + y_loss)
    return mutual_loss


def grad_loss(input_I_low, input_I_high):
    mse = nn.MSELoss()
    low_grad_x, low_grad_y = grad(input_I_low)
    high_grad_x, high_grad_y = grad(input_I_high)
    # x_loss = torch.abs(low_grad_x - high_grad_x)
    # y_loss = torch.abs(low_grad_y - high_grad_y)
    x_loss = mse(torch.abs(low_grad_x), torch.abs(high_grad_x))
    y_loss = mse(torch.abs(low_grad_y), torch.abs(high_grad_y))
    return x_loss+y_loss


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1: size // 2 + 1, -size // 2 + 1: size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    g = g / g.sum()
    g = torch.tensor(g)
    return g


def conv2_sep(f, sigma):
    ksize = round(5 * sigma) | 1
    g = fspecial_gauss(ksize, sigma)
    g = g.unsqueeze(0).unsqueeze(0)
    weight = nn.Parameter(data=g, requires_grad=False).type(dtype)
    ret = F.conv2d(f.unsqueeze(0), weight, stride=1, padding=ksize // 2, dilation=1, groups=1)
    return ret


def lpfilter(f, sigma):
    fb = f.clone()
    for i in range(f.shape[1]):
        fb[:, i] = conv2_sep(f[:, i], sigma)
    return fb



def RTV(f, sigma=3.0):
    fx, fy = grad(f)
    wto = torch.max(torch.sum(torch.sqrt(fx ** 2 + fy ** 2 + 1e-16), dim=1) / f.shape[1], torch.tensor(0.01)) ** -1
    fbin = lpfilter(f, sigma)
    gfx, gfy = grad(fbin)
    wtbx = torch.max(torch.sum(torch.abs(gfx), dim=1) / f.shape[1], torch.tensor(0.001)) ** -1
    wtby = torch.max(torch.sum(torch.abs(gfy), dim=1) / f.shape[1], torch.tensor(0.001)) ** -1
    retx = wtbx * wto
    rety = wtby * wto
    # print(retx.shape)
    retx[:, :, -1] = 0
    rety[:, -1, :] = 0
    return retx, rety


def RTV_loss(input_I_low):
    grd_x, grd_y = grad(input_I_low)
    wx, wy = RTV(input_I_low)
    return torch.sum(0.01 * (wx * (grd_x ** 2) + wy * (grd_y ** 2))) / input_I_low.shape[1]


def TV_loss(input_I_low):
    grd_x, grd_y = grad(input_I_low)
    grd = torch.sqrt(grd_x ** 2 + grd_y ** 2 + 1e-16)
    loss = torch.sum(grd) / input_I_low.shape[1]
    return loss


# input = torch.eye(5).unsqueeze(0).unsqueeze(0)
# input = input.repeat(1, 3, 1, 1)
# print(input.shape)
# print(grad(input))
