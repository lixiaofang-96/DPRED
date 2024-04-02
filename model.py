import torch
import torch.nn as nn
from myutils import *
import torch.nn.functional as F
import cv2
import torch.fft

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')

class HyPaNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=5, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus()).to(device)

    def forward(self, x):
        x = self.mlp(x) + 1
        return x


class HyParNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1):
        super(HyParNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_nc),
            nn.Sigmoid()).to(device)

    def forward(self, x):
        x = self.mlp(x) + 0.001
        return x


class DecomNet_RTV(nn.Module):
    def __init__(self, in_ch, k1=10):
        super(DecomNet_RTV, self).__init__()
        self.hypar = HyPaNet(1, k1).to(device)
        self.par1 = HyParNet(in_ch).to(device)
        self.par2 = HyParNet(in_ch).to(device)
        self.par3 = HyParNet(in_ch).to(device)
        self.k1 = k1

    def forward(self, O):
        batch, ch, row, col = O.shape

        I = O.clone()
        d1 = torch.zeros_like(I).to(device)
        d2 = torch.zeros_like(I).to(device)
        y1 = torch.zeros_like(I).to(device)
        y2 = torch.zeros_like(I).to(device)

        # mu = torch.ones([1, self.k1, 1, 1])
        # alpha = 0.001
        # px = 1
        # py = 1

        mu1 = torch.tensor(1.0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mu = self.hypar(mu1.to(device)).to(device)
        alpha = 0.001 * self.par1(O).to(device)
        px = self.par2(O).to(device)
        py = self.par3(O).to(device)

        eps = 0.001
        Dx = ([1.0], [-1.0])
        Dy = ([1.0, -1.0])
        Dx = torch.tensor(Dx).unsqueeze(0).unsqueeze(0).to(device)
        Dy = torch.tensor(Dy).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        eigDtD = torch.pow(torch.abs(fftn(Dx, batch, ch, col, row, 2)), 2) + torch.pow(
            torch.abs(fftnt(Dy, batch, ch, row, col, 3)), 2).to(device)

        I_list = []
        R_list = []

        for i in range(self.k1):
            rhs_I = O - mu[0, i, 0, 0] * Dive(d1 + y1, d2 + y2)
            lhs_I = 1 + mu[0, i, 0, 0] * eigDtD
            I = torch.real(torch.fft.ifftn(torch.fft.fftn(rhs_I) / lhs_I))
            DxI, DyI = grad(I)

            wtbx = torch.max(torch.pow(torch.abs(DxI.to(device)), (2 - px)), torch.tensor(eps).to(device)) ** (-1)
            wtby = torch.max(torch.pow(torch.abs(DyI.to(device)), (2 - py)), torch.tensor(eps).to(device)) ** (-1)
            wx = wtbx.to(device)
            wy = wtby.to(device)
            d1 = mu[0, i, 0, 0] * (DxI.to(device) - y1) / (2 * alpha * wx + mu[0, i, 0, 0])
            d2 = mu[0, i, 0, 0] * (DyI.to(device) - y2) / (2 * alpha * wy + mu[0, i, 0, 0])
            y1 = y1 + (d1 - DxI.to(device))
            y2 = y2 + (d2 - DyI.to(device))
            I_list.append(I)
            # wxx = torch.sqrt(alpha * wx)
            # wyy = torch.sqrt(alpha * wy)

        R = O / (I + eps)
        R_list.append(R)

        return I_list, R_list, alpha, px, py, mu


class ConvReLUBlock(nn.Module):

    def __init__(self):
        super(ConvReLUBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))


class EnhanceNet_I(nn.Module):

    def __init__(self, in_ch=6, out_ch=3):
        super(EnhanceNet_I, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu_down1 = nn.ReLU(inplace=True)
        self.layers1 = self.make_layer(ConvReLUBlock, 1)
        self.down2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu_down2 = nn.ReLU(inplace=True)
        self.layers2 = self.make_layer(ConvReLUBlock, 1)
        self.up1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_up1 = nn.ReLU(inplace=True)
        self.layers3 = self.make_layer(ConvReLUBlock, 1)
        self.up2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_up2 = nn.ReLU(inplace=True)
        self.output = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, x_ratio):
        # x = x ** (1 / 2.2)
        x1 = torch.cat([x, x_ratio], dim=1)
        y = self.input(x1)
        y1 = y
        y = self.relu_down1(self.down1(y))
        y = self.layers1(y)
        y2 = y
        y = self.relu_down2(self.down2(y))
        y = self.layers2(y)
        y = self.relu_up1(self.up1(y)) + y2
        y = self.layers3(y)
        y = self.relu_up2(self.up2(y)) + y1
        y = self.output(y)  # + x
        return y


if __name__ == '__main__':
    img = cv2.imread('/home/www/myRetinex/data/myLOL/eval15/high/V/1.png', cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('22', img)
    # cv2.waitKey(0)
    img = np.asarray(img) / 255.0
    print(img.shape)
    img = np.expand_dims(img, axis=2)
    # img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]

    input = torch.from_numpy(img).float()
    input = input.permute(2, 0, 1)
    input = input.unsqueeze(0)
    input = input.to(device)
    print(input.shape)
    model = DecomNet_RTV().cuda()
    # low_I_list, low_R_list, px, py, qx, qy, mu = model(input, alpha=0.001)
    low_I_list, low_R_list, alpha, px, py, mu = model(input)
    low_I = low_I_list[-1]
    low_R = low_R_list[-1]
    print(low_I.shape)
    print(low_R.shape)

    path_save = './net_results/'
    os.makedirs(path_save, exist_ok=True)

    u = low_I[0].cpu().detach().numpy() * 255.0
    u[u < 0] = 0
    u[u > 255] = 255
    u = np.transpose(u, (1, 2, 0))
    # u[:, :, [2, 1, 0]] = u[:, :, [0, 1, 2]]
    cv2.imwrite(path_save + 'ill1.png', u)

    # u = low_S[0].cpu().detach().numpy() * 255.0
    # u[u < 0] = 0
    # u[u > 255] = 255
    # u = np.transpose(u, (1, 2, 0))
    # u[:, :, [2, 1, 0]] = u[:, :, [0, 1, 2]]
    # cv2.imwrite(path_save + 'struct.png', u)
    #
    # u = low_T[0].cpu().detach().numpy() * 255.0
    # u[u < 0] = 0
    # u[u > 255] = 255
    # u = np.transpose(u, (1, 2, 0))
    # u[:, :, [2, 1, 0]] = u[:, :, [0, 1, 2]]
    # cv2.imwrite(path_save + 'text.png', u)

    v = low_R[0].cpu().detach().numpy() * 255.0
    v[v < 0] = 0
    v[v > 255] = 255
    v = np.transpose(v, (1, 2, 0))
    # u[:, :, [2, 1, 0]] = u[:, :, [0, 1, 2]]
    cv2.imwrite(path_save + 'reflect1.png', v)

    O = low_I * low_R
    O = O[0].cpu().detach().numpy() * 255.0
    O[O < 0] = 0
    O[O > 255] = 255
    O = np.transpose(O, (1, 2, 0))
    cv2.imwrite(path_save + 'recon1.png', O)
