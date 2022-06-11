import torch
import torch.nn as nn
import numpy as np
import math
import copy
import torch.nn.functional as F
from imresize import imresize, imresize_to_shape


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_activation(opt):
    activations = {"lrelu": nn.LeakyReLU(opt.lrelu_alpha, inplace=True),
                   "elu": nn.ELU(alpha=1.0, inplace=True),
                   "prelu": nn.PReLU(num_parameters=1, init=0.25),
                   "selu": nn.SELU(inplace=True)
                   }
    return activations[opt.activation]


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def upsample(x, size):
    # https://blog.csdn.net/moshiyaofei/article/details/102243913
    x_up = torch.nn.functional.interpolate(x, size=size, mode='trilinear', align_corners=True)
    return x_up


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, opt, generator=False):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv3d(in_channel, out_channel, kernel_size=ker_size, stride=1, padding=padd))
        if generator and opt.batch_norm:
            self.add_module('norm', nn.BatchNorm3d(out_channel))
        self.add_module(opt.activation, get_activation(opt))


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, opt)

        self.body = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt)
            self.body.add_module('block%d' % (i), block)

        self.tail = nn.Conv3d(N, 1, kernel_size=opt.ker_size, padding=opt.padd_size)

    def forward(self, x):
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


# class GrowingGenerator(nn.Module):
#     def __init__(self, opt):
#         super(GrowingGenerator, self).__init__()
#
#         self.opt = opt
#         N = int(opt.nfc)
#
#         self._pad = nn.ConstantPad3d(1, 0)
#         self._pad_block = nn.ConstantPad3d(opt.num_layer - 1, 0) if opt.train_mode == "generation" \
#                                                                     or opt.train_mode == "animation" \
#             else nn.ConstantPad3d(opt.num_layer, 0)
#
#         self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, opt, generator=True)
#
#         # add
#         self.res = nn.Sequential()
#         self.res.ca=ChannelAttention(N)
#         self.res.sa=SpatialAttention()
#         for i in range(opt.n_residual_blocks):
#             self.res.add_module('residual_block' + str(i + 1), residualBlock())
#         self.res.ca1=ChannelAttention(N)
#         self.res.sa1=SpatialAttention()
#
#         self.body = torch.nn.ModuleList([])
#         _first_stage = nn.Sequential()
#         for i in range(opt.num_layer):
#             block = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt, generator=True)
#             _first_stage.add_module('block%d' % (i), block)
#         self.body.append(_first_stage)
#
#
#         self.tail = nn.Sequential(
#             nn.Conv3d(N, opt.nc_im, kernel_size=opt.ker_size, padding=opt.padd_size),
#             nn.Tanh())
#
#     def init_next_stage(self):
#         self.body.append(copy.deepcopy(self.body[-1]))
#
#     def forward(self, noise, real_shapes, noise_amp):
#         x = self.head(self._pad(noise[0]))
#
#         x = self.res.ca(x) * x
#         x = self.res.sa(x) * x
#
#         for i in range(self.opt.n_residual_blocks):
#             x = self.res.__getattr__('residual_block' + str(i+1))(x)
#
#         x = self.res.ca1(x) * x
#         x = self.res.sa1(x) * x
#
#         # we do some upsampling for training models for unconditional generation to increase
#         # the image diversity at the edges of generated images
#         if self.opt.train_mode == "generation" or self.opt.train_mode == "animation":
#             x = upsample(x, size=[x.shape[2] + 2, x.shape[3] + 2, x.shape[4] + 2])
#         x = self._pad_block(x)
#         x_prev_out = self.body[0](x)
#
#         for idx, block in enumerate(self.body[1:], 1):
#             if self.opt.train_mode == "generation" or self.opt.train_mode == "animation":
#                 x_prev_out_1 = upsample(x_prev_out,
#                                         size=[real_shapes[idx][2], real_shapes[idx][3], real_shapes[idx][4]])
#                 x_prev_out_2 = upsample(x_prev_out, size=[real_shapes[idx][2] + self.opt.num_layer * 2,
#                                                           real_shapes[idx][3] + self.opt.num_layer * 2,
#                                                           real_shapes[idx][4] + self.opt.num_layer * 2])
#                 x_prev = block(x_prev_out_2 + noise[idx] * noise_amp[idx])
#             else:
#                 x_prev_out_1 = upsample(x_prev_out, size=real_shapes[idx][2:])
#                 x_prev = block(self._pad_block(x_prev_out_1 + noise[idx] * noise_amp[idx]))
#             x_prev_out = x_prev + x_prev_out_1
#
#         #out = self.tail(self._pad(x_prev_out))
#         out = self.tail(self._pad(x_prev_out))
#         return out

#添加CBAM的模块
class GrowingGenerator(nn.Module):
    def __init__(self, opt):
        super(GrowingGenerator, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        self._pad = nn.ConstantPad3d(1, 0)
        self._pad_block = nn.ConstantPad3d(opt.num_layer - 1, 0) if opt.train_mode == "generation" \
                                                                    or opt.train_mode == "animation" \
            else nn.ConstantPad3d(opt.num_layer, 0)

        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, opt, generator=True)

        self.res = nn.Sequential()
        for i in range(opt.n_residual_blocks):
            self.res.add_module('residual_block' + str(i + 1), residualBlock())

        self.body = torch.nn.ModuleList([])
        _first_stage = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt, generator=True)
            _first_stage.add_module('block%d' % (i), block)
        self.body.append(_first_stage)


        self.tail = nn.Sequential(
            nn.Conv3d(N, opt.nc_im, kernel_size=opt.ker_size, padding=opt.padd_size),
            nn.Tanh())

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise, real_shapes, noise_amp):
        x = self.head(self._pad(noise[0]))

        for i in range(self.opt.n_residual_blocks):
            x = self.res.__getattr__('residual_block' + str(i+1))(x)


        # we do some upsampling for training models for unconditional generation to increase
        # the image diversity at the edges of generated images
        if self.opt.train_mode == "generation" or self.opt.train_mode == "animation":
            x = upsample(x, size=[x.shape[2] + 2, x.shape[3] + 2, x.shape[4] + 2])
        x = self._pad_block(x)
        x_prev_out = self.body[0](x)

        for idx, block in enumerate(self.body[1:], 1):
            if self.opt.train_mode == "generation" or self.opt.train_mode == "animation":
                x_prev_out_1 = upsample(x_prev_out,
                                        size=[real_shapes[idx][2], real_shapes[idx][3], real_shapes[idx][4]])
                x_prev_out_2 = upsample(x_prev_out, size=[real_shapes[idx][2] + self.opt.num_layer * 2,
                                                          real_shapes[idx][3] + self.opt.num_layer * 2,
                                                          real_shapes[idx][4] + self.opt.num_layer * 2])
                x_prev = block(x_prev_out_2 + noise[idx] * noise_amp[idx])
            else:
                x_prev_out_1 = upsample(x_prev_out, size=real_shapes[idx][2:])
                x_prev = block(self._pad_block(x_prev_out_1 + noise[idx] * noise_amp[idx]))
            x_prev_out = x_prev + x_prev_out_1

        out = self.tail(self._pad(x_prev_out))
        return out


def swish(x):
    return x * torch.sigmoid(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, n, k, stride=s, padding=1)
        self.conv2 = nn.Conv3d(n, n, k, stride=s, padding=1)

        self.cbam = CBAM(channel=64)

    def forward(self, x):
        # y = swish(self.conv1(x))
        # return self.conv2(y) + x
        y = swish(self.conv1(x))
        y2 = self.conv2(y)
        out = self.cbam(y2)
        return out + x

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

# class ResBlock_CBAM(nn.Module):
#     def __init__(self,in_places=64, places=64, stride=1,downsampling=False, expansion = 1):
#         super(ResBlock_CBAM,self).__init__()
#         self.expansion = expansion
#         self.downsampling = downsampling
#
#         self.bottleneck = nn.Sequential(
#             nn.Conv3d(in_channels=in_places,out_channels=places,kernel_size=3,stride=1, bias=False),
#             #nn.BatchNorm3d(places),
#             #nn.ReLU(inplace=True),
#             nn.Conv3d(in_channels=places, out_channels=places, kernel_size=3, stride=1, padding=1, bias=False),
#             #nn.BatchNorm3d(places),
#             #nn.ReLU(inplace=True),
#             #nn.Conv3d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
#             #nn.BatchNorm3d(places*self.expansion),
#         )
#
#         self.cbam = CBAM(channel=places*self.expansion)
#
#         if self.downsampling:
#             self.downsample = nn.Sequential(
#                 nn.Conv3d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm3d(places*self.expansion)
#             )
#         #self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         residual = x
#         out = self.bottleneck(x)
#         out = self.cbam(out)
#         if self.downsampling:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#         return out


#CBAM-BAM
# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)
#
# class ChannelGate(nn.Module):
#     def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
#         super(ChannelGate, self).__init__()
#         self.gate_c = nn.Sequential()
#         self.gate_c.add_module('flatten', Flatten())
#
#         gate_channels = [gate_channel]  # eg 64
#         gate_channels += [gate_channel // reduction_ratio] * num_layers  # eg 4
#         gate_channels += [gate_channel]  # 64
#         #print(gate_channels)
#         # gate_channels: [64, 4, 4]
#
#         for i in range(len(gate_channels) - 2):
#             self.gate_c.add_module(
#                 'gate_c_fc_%d' % i,
#                 nn.Linear(gate_channels[i], gate_channels[i + 1]))
#             self.gate_c.add_module('gate_c_bn_%d' % (i + 1),
#                                    nn.BatchNorm3d(gate_channels[i + 1]))
#             self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
#
#         self.gate_c.add_module('gate_c_fc_final',
#                                nn.Linear(gate_channels[-2], gate_channels[-1]))
#
#     def forward(self, x):
#         avg_pool = F.avg_pool3d(x, x.size(2), stride=x.size(2))
#         #print(avg_pool.shape)
#         return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)
#
# class SpatialGate(nn.Module):
#     def __init__(self,
#                  gate_channel,
#                  reduction_ratio=16,
#                  dilation_conv_num=2,
#                  dilation_val=4):
#         super(SpatialGate, self).__init__()
#         self.gate_s = nn.Sequential()
#
#         self.gate_s.add_module(
#             'gate_s_conv_reduce0',
#             nn.Conv3d(gate_channel,
#                       gate_channel // reduction_ratio,
#                       kernel_size=1))
#         self.gate_s.add_module('gate_s_bn_reduce0',
#                                nn.BatchNorm3d(gate_channel // reduction_ratio))
#         self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
#
#         # 进行多个空洞卷积，丰富感受野
#         for i in range(dilation_conv_num):
#             self.gate_s.add_module(
#                 'gate_s_conv_di_%d' % i,
#                 nn.Conv3d(gate_channel // reduction_ratio,
#                           gate_channel // reduction_ratio,
#                           kernel_size=3,
#                           padding=dilation_val,
#                           dilation=dilation_val))
#             self.gate_s.add_module(
#                 'gate_s_bn_di_%d' % i,
#                 nn.BatchNorm3d(gate_channel // reduction_ratio))
#             self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
#
#         self.gate_s.add_module(
#             'gate_s_conv_final',
#             nn.Conv3d(gate_channel // reduction_ratio, 1, kernel_size=1))
#
#     def forward(self, x):
#         return self.gate_s(x).expand_as(x)
#
# class BAM(nn.Module):
#     def __init__(self, gate_channel):
#         super(BAM, self).__init__()
#         self.channel_att = ChannelGate(gate_channel)
#         self.spatial_att = SpatialGate(gate_channel)
#
#     def forward(self, x):
#         att = 1 + F.sigmoid(self.channel_att(x) * self.spatial_att(x))
#         return att * x