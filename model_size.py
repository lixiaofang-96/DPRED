from thop import profile
from model import DecomNet_RTV, EnhanceNet_I
from network_unet import UNetRes as net
import torch
from thop import clever_format
from network_ffdnet import FFDNet as ffdnet
from DnCNN import DnCNN
from network_scunet import SCUNet as scunet
from network_dncnn import DnCNN as dncnn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Decom_net = DecomNet_RTV(in_ch=1, k1=10).cuda()
Enhance_I = EnhanceNet_I(in_ch=2, out_ch=1).cuda()
Enhance_R = EnhanceNet_I(in_ch=2, out_ch=1).cuda()
n_channels = 3
noise_level_img = 15  # set AWGN noise level for noisy image
noise_level_model = noise_level_img  # set noise level for model
Denoise_enI = net(in_nc=n_channels + 1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                  downsample_mode="strideconv", upsample_mode="convtranspose").cuda()

input = torch.randn(1, 1, 400, 600).to(device)
Decom_net_macs, Decom_net_params = profile(Decom_net, inputs=(input,))
Enhance_I_macs, Enhance_I_params = profile(Enhance_I, inputs=(input, input))
Enhance_R_macs, Enhance_R_params = profile(Enhance_I, inputs=(input, input))
input1 = torch.randn(1, 4, 400, 600).to(device)
# sigma = torch.full((1, 1, 1, 1), 15 / 255.).type_as(input)
Denoise_enI_macs, Denoise_enI_params = profile(Denoise_enI, inputs=(input1,))

total_macs = Decom_net_macs + Enhance_I_macs + Enhance_R_macs + Denoise_enI_macs
total_params = Decom_net_params + Enhance_I_params + Enhance_R_params + Denoise_enI_params

Decom_net_macs, Decom_net_params = clever_format([Decom_net_macs, Decom_net_params], "%.5f")
Enhance_I_macs, Enhance_I_params = clever_format([Enhance_I_macs, Enhance_I_params], "%.5f")
Enhance_R_macs, Enhance_R_params = clever_format([Enhance_R_macs, Enhance_R_params], "%.5f")
Denoise_enI_macs, Denoise_enI_params = clever_format([Denoise_enI_macs, Denoise_enI_params], "%.5f")
total_macs, total_params = clever_format([total_macs, total_params], "%.5f")

print(Decom_net_macs, Decom_net_params)
print('--------------------------------')
print(Enhance_I_macs, Enhance_I_params)
print('--------------------------------')
print(Enhance_R_macs, Enhance_R_params)
print('--------------------------------')
print(Denoise_enI_macs, Denoise_enI_params)
print('--------------------------------')
print(total_macs, total_params)
