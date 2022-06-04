import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import fpn
import utils

import fusion_strategy

#上采样操作
class UpsampleReshape_eval(torch.nn.Module):   #继承torch.nn.Module
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__() #初始化
        # self.up = nn.Upsample(scale_factor=2) #增加了一个up操作，2倍上采样
        self.up = F.interpolate

    def forward(self, x1, x2):
        x2 = self.up(x2,scale_factor=2)       #对x2进行上采样
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        #对上采样后特征图的大小进行填补，保证它和别的特征图大小相同
        if shape_x1[3] != shape_x2[3]:        #N,C,H,W
            lef_right = shape_x1[3] - shape_x2[3]          #宽之差
            if lef_right%2 is 0.0:        #宽之差是偶数
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:   #高之差
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]     #填充操作
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2

class ConvTranspose2d_eval(torch.nn.Module):   #继承torch.nn.Module
    def __init__(self):
        super(ConvTranspose2d_eval, self).__init__() #初始化
        # self.up = nn.Upsample(scale_factor=2) #增加了一个up操作，2倍上采样

    def forward(self, x1, x2,nb_filter, output_nc, kernel_size, stride, padding, bias):
        x2 = self.up(x2,scale_factor=2)       #对x2进行上采样
        x2=nn.ConvTranspose2d(nb_filter, output_nc, kernel_size, stride, padding, bias)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        #对上采样后特征图的大小进行填补，保证它和别的特征图大小相同
        if shape_x1[3] != shape_x2[3]:        #N,C,H,W
            lef_right = shape_x1[3] - shape_x2[3]          #宽之差
            if lef_right%2 is 0.0:        #宽之差是偶数
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:   #高之差
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]     #填充操作
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):  #输入通道数，输出通道数，卷积核大小，步长，是否是最后一层
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))    #卷积前后特征图大小不变
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5) #dropout防止过拟合
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)   #填边
        out = self.conv2d(out)          #卷积
        if self.is_last is False:
            out = F.relu(out, inplace=True)  #inplace代表直接赋值
        return out


# light version
class DenseBlock_light(torch.nn.Module):        #两个卷积层，先将通道缩小一半，再按给的输出通道数输出
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        out_channels_def = int(in_channels / 2)     #输出通道数为输入通道数的一半
        # out_channels_def = out_channels
        denseblock = []

        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

class RXDBlock(torch.nn.Module):        #RXDN的块作为编码器的编码块
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(RXDBlock, self).__init__()
        out_channels_def = int(out_channels / 2)
        # out_channels_def = int(out_channels)
        self.RXDD_1 = ConvLayer(in_channels, out_channels_def, 1, stride)
        self.RXDD_2 = ConvLayer(out_channels_def, out_channels_def, kernel_size, stride)
        self.RXDD_3 = ConvLayer(out_channels_def * 2, out_channels_def, kernel_size, stride)
        self.RXDD_4 = ConvLayer(out_channels_def * 3, out_channels_def, kernel_size, stride)
        self.RXDD_5 = ConvLayer(out_channels_def * 4, out_channels, 1, stride, True)

        self.RXDR_1 = ConvLayer(in_channels, out_channels_def, 1, stride)
        self.RXDR_2 = ConvLayer(out_channels_def, out_channels_def, kernel_size, stride)
        self.RXDR_3 = ConvLayer(out_channels_def, out_channels, 1, stride)

    def forward(self, x):
        x1 = self.RXDR_3(self.RXDR_2(self.RXDR_1(x)))
        x2_1 = self.RXDD_1(x)
        x2_2 = self.RXDD_2(x2_1)
        x2_3 = self.RXDD_3(torch.cat([x2_1, x2_2], 1))
        x2_4 = self.RXDD_4(torch.cat([x2_1, x2_2, x2_3],1))
        x2 = self.RXDD_5(torch.cat([x2_1, x2_2, x2_3, x2_4],1))
        out = x1 + x2 + x
        # out = x + x2
        return out
# class RXDBlock(torch.nn.Module):  # RXDN的块作为编码器的编码块
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super(RXDBlock, self).__init__()
#             # out_channels_def = int(out_channels / 2)
#         out_channels_def = int(out_channels)
#
#         self.RXDD_1 = ConvLayer(out_channels_def, out_channels_def, kernel_size, stride)
#         self.RXDD_2 = ConvLayer(out_channels_def * 2, out_channels_def, kernel_size, stride)
#         self.RXDD_3 = ConvLayer(out_channels_def * 3, out_channels_def, kernel_size, stride)
#         self.RXDD_4 = ConvLayer(out_channels_def * 4, out_channels_def, 1, stride)
#
#         self.RXDR_1 = ConvLayer(out_channels_def, out_channels_def, kernel_size, stride)
#
#     def forward(self, x):
#         x1 = self.RXDR_1(x)
#         x2_1 = self.RXDD_1(x)
#         x2_2 = self.RXDD_2(torch.cat([x2_1, x], 1))
#         x2_3 = self.RXDD_3(torch.cat([x2_1, x2_2, x],1))
#         x2 = self.RXDD_4(torch.cat([x2_1, x2_2, x2_3, x],1))
#         out = x1 + x2 + x
#         return out


# NestFuse network - light, no desnse
class NestFuse_autoencoder(nn.Module):          #自编码器
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
        super(NestFuse_autoencoder, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 64
        kernel_size = 3
        stride = 1

        self.pool1 = nn.MaxPool2d(2, 2)         #最大池化
        self.pool2 = nn.MaxPool2d(4, 4)
        self.pool3 = nn.MaxPool2d(8, 8)
        # self.up = nn.Upsample(scale_factor=2)       #上采样
        self.up = F.interpolate
        self.up_eval = UpsampleReshape_eval()        #
        # self.fpn=fpn.FPN101()
        self.convtranspose=ConvTranspose2d_eval()

        # encoder
        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)           #卷积操作。输入通道数。输出通道数，卷积核大小，步长
        self.DB1_0 = RXDBlock(output_filter, nb_filter[0], kernel_size, 1)        #实例化block
        self.DB2_0 = RXDBlock(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0 = RXDBlock(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0 = RXDBlock(nb_filter[2], nb_filter[3], kernel_size, 1)
        # self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)  # 实例化block
        # self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        # self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        # self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)

        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)

        # self.DB_all = ConvLayer(nb_filter[0] * 3, nb_filter[0], 1, stride)

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)
            # self.conv_out =nn.ConvTranspose2d(nb_filter[0], output_nc, kernel_size=4, stride=2, padding=1, bias=False)
            # self.conv_out_eval = self.convtranspose(x1,x2,nb_filter[0], output_nc, kernel_size=4, stride=2, padding=1, bias=False)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=False) + y

    def _subsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=False) + y

    def encoder(self, input):
        x = self.conv0(input)
        # utils.save_image_test_temp(x , './outputs/1.png')

        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool1(x1_0))
        x3_0 = self.DB3_0(self.pool1(x2_0))
        x4_0 = self.DB4_0(self.pool1(x3_0))
        # utils.save_image_test_temp(x1_0, './outputs/1.png')
        # utils.save_image_test_temp(x2_0, './outputs/2.png')
        # utils.save_image_test_temp(x3_0, './outputs/3.png')
        # utils.save_image_test_temp(x4_0, './outputs/4.png')
        #
        # x4 = x4_0
        # x3 = self._upsample_add(x4, x3_0)
        # x2 = self._upsample_add(x3, x2_0)
        # x1 = self._upsample_add(x2, x1_0)

        # return [x1, x2, x3, x4]
        return [x1_0, x2_0, x3_0, x4_0]

    def fusion(self, en1, en2, p_type):
        # attention weight
        # fusion_function1 = fusion_strategy.attention_fusion_weight1
        fusion_function1 = fusion_strategy.attention_fusion_weight1

        f1_0 = fusion_function1(en1[0], en2[0], p_type)
        # utils.save_image_test_temp(f1_0, './outputs/2.png')
        f2_0 = fusion_function1(en1[1], en2[1], p_type)
        # utils.save_image_test_temp(f2_0, './outputs/3.png')
        f3_0 = fusion_function1(en1[2], en2[2], p_type)
        # utils.save_image_test_temp(f3_0, './outputs/4.png')
        f4_0 = fusion_function1(en1[3], en2[3], p_type)
        # utils.save_image_test_temp(f4_0, './outputs/5.png')

        # utils.save_image_test_temp(en1[0], './outputs/1.1.png')
        # utils.save_image_test_temp(en1[1], './outputs/2.1.png')
        # utils.save_image_test_temp(en1[2], './outputs/3.1.png')
        # utils.save_image_test_temp(en1[3], './outputs/4.1.png')
        # utils.save_image_test_temp(en2[0], './outputs/1.2.png')
        # utils.save_image_test_temp(en2[1], './outputs/2.2.png')
        # utils.save_image_test_temp(en2[2], './outputs/3.2.png')
        # utils.save_image_test_temp(en2[3], './outputs/4.2.png')
        # utils.save_image_test_temp(f1_0, './outputs/1.png')
        # utils.save_image_test_temp(f2_0, './outputs/2.png')
        # utils.save_image_test_temp(f3_0, './outputs/3.png')
        # utils.save_image_test_temp(f4_0, './outputs/4.png')


        return [f1_0, f2_0, f3_0, f4_0]

    def decoder_train(self, f_en):
        # x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1],scale_factor=2)], 1))    #在1维即通道维度进行级联，传的参数不太对？
        #
        # x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2],scale_factor=2)], 1))
        # x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1,scale_factor=2)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3],scale_factor=2)], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], self.up(x3_1,scale_factor=2)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], self.up(x2_2,scale_factor=2)], 1))

        if self.deepsupervision:          #没看懂
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            # output = self.conv_out(x1_3)
            output = self.conv_out(x1_3)
            return [output]

    def decoder_eval(self, f_en):

        # x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))
        #
        # x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        # x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        # utils.save_image_test_temp(x3_1, './outputs/6.png')
        x2_2 = self.DB2_2(torch.cat([f_en[1], self.up_eval(f_en[1], x3_1)], 1))
        # utils.save_image_test_temp(x2_2, './outputs/7.png')
        x1_3 = self.DB1_3(torch.cat([f_en[0], self.up_eval(f_en[0], x2_2)], 1))
        # utils.save_image_test_temp(x1_3, './outputs/8.png')
        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            # output = self.conv_out_eval(img,self.up(x1_3,scale_factor=2))
            return [output]