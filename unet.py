
import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
    
class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


def upsize(x,scale_factor=2):
    #x = F.interpolate(x, size=e.shape[2:], mode='nearest')
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x

class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=256,
                 kernel_size=3,
                 is_deconv=False,
                 ):
        super().__init__()
 
        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)
 
        # B, C/4, H, W -> B, C/4, H, W
        '''
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=conv_padding,bias=False)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2,**up_kwargs)
        '''
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels // 2)
        self.relu2 = nn.ReLU(inplace=True)
 
        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels , kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d( out_channels)
        self.relu3 = nn.ReLU(inplace=True)
 
    def forward(self, x):
        x = torch.cat(x,1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
 
 
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class ResNet34UnetPlus(nn.Module):
    def __init__(self,
                 num_class,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 ):
        super().__init__()
 
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.base_size=512
        self.crop_size=512
        self._up_kwargs={'mode': 'bilinear', 'align_corners': True}
        
        
        self.mix = nn.Parameter(torch.FloatTensor(5))
        self.mix.data.fill_(1)
        
        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),bias=False)
 
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
 
        # Decoder
        self.decoder0_1 =  DecoderBlock(in_channels=64+64,
                                   out_channels=64,
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)


        self.decoder1_1 = DecoderBlock(in_channels=128+64,
                                   out_channels=64,
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder0_2 =  DecoderBlock(in_channels=64+64+64,
                                   out_channels=64,
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)


        self.decoder2_1 = DecoderBlock(in_channels=128+256,
                                   out_channels=128,
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder1_2 = DecoderBlock(in_channels=64+64+128,
                                   out_channels=128,
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder0_3 = DecoderBlock(in_channels=64+64+64+128,
                                   out_channels=128,
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)


        self.decoder3_1 = DecoderBlock(in_channels=512+256,
                                   out_channels=256,
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder2_2 = DecoderBlock(in_channels=128+128+256,
                                   out_channels=256,
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder1_3 = DecoderBlock(in_channels=64+64+128+256,
                                   out_channels=256,
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder0_4 = DecoderBlock(in_channels=64+64+64+128+256,
                                   out_channels=256,
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)

        self.logit1 = nn.Conv2d( 64,num_class, kernel_size=1)
        self.logit2 = nn.Conv2d( 64,num_class, kernel_size=1)
        self.logit3 = nn.Conv2d(128,num_class, kernel_size=1)
        self.logit4 = nn.Conv2d(256,num_class, kernel_size=1)
        self.outlayer = nn.Linear(224*224,8)

 
    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]
 
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad
 
    def forward(self, x):
        _,_, H,W = x.shape
        # stem
        x = self.firstconv(x)   #subsample
        x = self.firstbn(x)
        x_ = self.firstrelu(x)  

        # Encoder
        x = self.firstmaxpool(x_) #64
        e1 = self.encoder1(x)  #64
        e2 = self.encoder2(e1)  #128
        e3 = self.encoder3(e2)  #256
        e4 = self.encoder4(e3)  #512
 #--------Unet Plus Plus Decoder----------------------------------------------

        x0_0 = x_
        x1_0 = e1
        # print(x0_0.shape, x1_0.shape)  #64 128 128
        x0_1 = self.decoder0_1([x0_0, upsize(x1_0)])  # 256 256

        x2_0 = e2
        x1_1 = self.decoder1_1([x1_0, upsize(x2_0)])
        # print(x0_0.shape, x0_1.shape, x1_1.shape)
        x0_2 = self.decoder0_2([x0_0, x0_1,  upsize(x1_1)])

        x3_0 = e3
        x2_1 = self.decoder2_1([x2_0, upsize(x3_0)])
        x1_2 = self.decoder1_2([x1_0, x1_1, upsize(x2_1)])
        x0_3 = self.decoder0_3([x0_0, x0_1, x0_2,  upsize(x1_2)])

        x4_0 = e4
        x3_1 = self.decoder3_1([x3_0, upsize(x4_0)])
        x2_2 = self.decoder2_2([x2_0, x2_1, upsize(x3_1)])
        x1_3 = self.decoder1_3([x1_0, x1_1, x1_2, upsize(x2_2)])
        x0_4 = self.decoder0_4([x0_0, x0_1, x0_2,  x0_3,  upsize(x1_3)])


        logit1 = self.logit1(x0_1)
        logit2 = self.logit2(x0_2)
        logit3 = self.logit3(x0_3)
        logit4 = self.logit4(x0_4)
        # print(self.mix)
        logit = self.mix[1]*logit1 + self.mix[2]*logit2 + self.mix[3]*logit3 + self.mix[4]*logit4
        logit = F.interpolate(logit, size=(H,W), mode='bilinear', align_corners=False)
        logit = self.outlayer(logit.squeeze(1).flatten(1,-1))


        return logit


class UNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        #
        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        # # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        #
        self.outconv1 = nn.Conv2d(filters[0], 1, 3, padding=1)
        self.outlayer = nn.Linear(224*224,8)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        d1 = self.outconv1(up1)  # 256
        d1 = self.outlayer(d1.squeeze(1).flatten(1,-1))

        return d1



if __name__=='__main__':
    img = torch.rand((16,3,224,224))
    model = UNet()
    out = model(img)
    print(out)