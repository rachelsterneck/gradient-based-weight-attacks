import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo

import torchvision

resnet18 = {
    '1':0, '2':1, '3':2, '4':4, '5':5, '6':7, '7':8, '8':10, '9':11,
    '10':13, '11':14, '12':16, '13':17, '14':19, '15':20, '16':22, '17':23
}
class VGG19(nn.Module):

    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        size_1 = 64
        size_2 = 64
        size_4 = 128
        size_5 = 128
        size_7 = 256
        size_8 = 256
        size_9 = 256
        size_10 = 256
        size_12 = 512
        size_13 = 512
        size_14 = 512
        size_15 = 512
        size_17 = 512
        size_18 = 512
        size_19 = 512
        size_20 = 512
        # 3 input channels for CIFAR10, VGG11 calls for 64 output channels from 
        # the first conv layer, a batchnorm, then a ReLU
        self.conv1 = nn.Conv2d(3, size_1, kernel_size = 3, padding = 1)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        #layer 2 is a conv that produces 64 channels, same format as layer 1
        self.conv2 = nn.Conv2d(size_1, size_2, kernel_size = 3, padding = 1)
        self.norm2 = nn.BatchNorm2d(size_1)
        self.relu2 = nn.ReLU()
        
        #layer 3 is a pooling layer
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        #layer 4 is a conv that produces 128 channels, same format as layer 1
        self.conv4 = nn.Conv2d(size_2, size_4, kernel_size = 3, padding = 1)
        self.norm4 = nn.BatchNorm2d(size_4)
        self.relu4 = nn.ReLU()
        
        #layer 5 is a conv that produces 128 channels, same format as layer 1
        self.conv5 = nn.Conv2d(size_4, size_5, kernel_size = 3, padding = 1)
        self.norm5 = nn.BatchNorm2d(size_5)
        self.relu5 = nn.ReLU()
        
        #layer 6 is a pooling layer
        self.pool6 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        #layer 7 is a conv that produces 256 channels, same format as layer 1
        self.conv7 = nn.Conv2d(size_5, size_7, kernel_size = 3, padding = 1)
        self.norm7 = nn.BatchNorm2d(size_7)
        self.relu7 = nn.ReLU()
        
        #layer 8 is a conv that produces 256 channels, same format as layer 1
        self.conv8 = nn.Conv2d(size_7, size_8, kernel_size = 3, padding = 1)
        self.norm8 = nn.BatchNorm2d(size_8)
        self.relu8 = nn.ReLU()
        
        #layer 9 is a conv that produces 256 channels, same format as layer 1
        self.conv9 = nn.Conv2d(size_8, size_9, kernel_size = 3, padding = 1)
        self.norm9 = nn.BatchNorm2d(size_9)
        self.relu9 = nn.ReLU()
        
        #layer 10 is a conv that produces 256 channels, same format as layer 1
        self.conv10 = nn.Conv2d(size_9, size_10, kernel_size = 3, padding = 1)
        self.norm10 = nn.BatchNorm2d(size_10)
        self.relu10 = nn.ReLU()
        
        #layer 11 is a pooling layer
        self.pool11 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        #layer 12 is a conv that produces 512 channels, same format as layer 1
        self.conv12 = nn.Conv2d(size_10, size_12, kernel_size = 3, padding = 1)
        self.norm12 = nn.BatchNorm2d(size_12)
        self.relu12 = nn.ReLU()
        
        #layer 13 is a conv that produces 512 channels, same format as layer 1
        self.conv13 = nn.Conv2d(size_12, size_13, kernel_size = 3, padding = 1)
        self.norm13 = nn.BatchNorm2d(size_13)
        self.relu13 = nn.ReLU()
        
        #layer 14 is a conv that produces 512 channels, same format as layer 1
        self.conv14 = nn.Conv2d(size_13, size_14, kernel_size = 3, padding = 1)
        self.norm14 = nn.BatchNorm2d(size_14)
        self.relu14 = nn.ReLU()
        
        #layer 15 is a conv that produces 512 channels, same format as layer 1
        self.conv15 = nn.Conv2d(size_14, size_15, kernel_size = 3, padding = 1)
        self.norm15 = nn.BatchNorm2d(size_15)
        self.relu15 = nn.ReLU()
        
        #layer 16 is a pooling layer
        self.pool16 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        #layer 17 is a conv that produces 512 channels, same format as layer 1
        self.conv17 = nn.Conv2d(size_15, size_17, kernel_size = 3, padding = 1)
        self.norm17 = nn.BatchNorm2d(size_17)
        self.relu17 = nn.ReLU()
        
        #layer 18 is a conv that produces 512 channels, same format as layer 1
        self.conv18 = nn.Conv2d(size_17, size_18, kernel_size = 3, padding = 1)
        self.norm18 = nn.BatchNorm2d(size_18)
        self.relu18 = nn.ReLU()
        
        #layer 19 is a conv that produces 512 channels, same format as layer 1
        self.conv19 = nn.Conv2d(size_18, size_19, kernel_size = 3, padding = 1)
        self.norm19 = nn.BatchNorm2d(size_19)
        self.relu19 = nn.ReLU()
        
        #layer 20 is a conv that produces 512 channels, same format as layer 1
        self.conv20 = nn.Conv2d(size_19, size_20, kernel_size = 3, padding = 1)
        self.norm20 = nn.BatchNorm2d(size_20)
        self.relu20 = nn.ReLU()
        
        #layer 21 is a pooling layer
        self.pool21 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        #layer 22 is an average pooling layer
        self.pool22 = nn.AvgPool2d(kernel_size=1, stride=1)
        
        #layer 23 is a fully connected layer
        self.full23 = nn.Linear(size_20, num_classes)

    def forward(self, x0):
        x1 = self.conv1(x0)
        x1 = self.norm1(x1)
        x1 = self.relu1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.norm2(x2)
        x2 = self.relu2(x2)
        
        x3 = self.pool3(x2)
        
        x4 = self.conv4(x3)
        x4 = self.norm4(x4)
        x4 = self.relu4(x4)
        
        x5 = self.conv5(x4)
        x5 = self.norm5(x5)
        x5 = self.relu5(x5)
        
        x6 = self.pool6(x5)
        
        x7 = self.conv7(x6)
        x7 = self.norm7(x7)
        x7 = self.relu7(x7)
        
        x8 = self.conv8(x7)
        x8 = self.norm8(x8)
        x8 = self.relu8(x8)
        
        x9 = self.conv9(x8)
        x9 = self.norm9(x9)
        x9 = self.relu9(x9)
        
        x10 = self.conv10(x9)
        x10 = self.norm10(x10)
        x10 = self.relu10(x10)
        
        x11 = self.pool11(x10)
        
        x12 = self.conv12(x11)
        x12 = self.norm12(x12)
        x12 = self.relu12(x12)
        
        x13 = self.conv13(x12)
        x13 = self.norm13(x13)
        x13 = self.relu13(x13)
        
        x14 = self.conv14(x13)
        x14 = self.norm14(x14)
        x14 = self.relu14(x14)
        
        x15 = self.conv15(x14)
        x15 = self.norm15(x15)
        x15 = self.relu15(x15)
        
        x16 = self.pool16(x15)
        
        x17 = self.conv17(x16)
        x17 = self.norm17(x17)
        x17 = self.relu17(x17)
        
        x18 = self.conv18(x17)
        x18 = self.norm18(x18)
        x18 = self.relu18(x18)
        
        x19 = self.conv19(x18)
        x19 = self.norm19(x19)
        x19 = self.relu19(x19)
        
        x20 = self.conv20(x19)
        x20 = self.norm20(x20)
        x20 = self.relu20(x20)
       
        x21 = self.pool21(x20)
        
        x22 = self.pool22(x21)
        
        x22 = x22.view(x20.size(0), -1)
        x23 = self.full23(x22)
        
        #return the activations from each layer as well as the output
        output = x23
        activations = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20]
        return output


class ResNet18(nn.Module):

    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        expansion = 1
        size_1 = 64   # 32 x 32
        size_2 = 64
        size_3 = 64
        size_4 = 64
        size_5 = 64
        size_6 = 128  # 16 x 16
        size_7 = 128
        size_8 = 128
        size_9 = 128
        size_10 = 256 # 8 x 8
        size_11 = 256
        size_12 = 256
        size_13 = 256
        size_14 = 512 # 4 x 4
        size_15 = 512
        size_16 = 512
        size_17 = 512
        self.conv1 = nn.Conv2d(3, size_1, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm1 = nn.BatchNorm2d(size_1)

        # BLOCK 1 #
        self.conv2 = nn.Conv2d(size_1, size_2, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm2 = nn.BatchNorm2d(size_2)
        self.conv3 = nn.Conv2d(size_2, size_3, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm3 = nn.BatchNorm2d(size_3)
        self.shortcut1 = nn.Conv2d(size_1, size_3, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.normS1 = nn.BatchNorm2d(size_3)

        self.conv4 = nn.Conv2d(size_3, size_4, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm4 = nn.BatchNorm2d(size_4)
        self.conv5 = nn.Conv2d(size_4, size_5, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm5 = nn.BatchNorm2d(size_5)
        self.shortcut2 = nn.Conv2d(size_3, size_5, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.normS2 = nn.BatchNorm2d(size_5)


        # BLOCK 2 #
        self.conv6 = nn.Conv2d(size_5, size_6, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.norm6 = nn.BatchNorm2d(size_6)
        self.conv7 = nn.Conv2d(size_6, size_7, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm7 = nn.BatchNorm2d(size_7)
        self.shortcut3 = nn.Conv2d(size_5, size_7, kernel_size = 1, stride = 2, padding = 0, bias = False)
        self.normS3 = nn.BatchNorm2d(size_7)
        
        self.conv8 = nn.Conv2d(size_7, size_8, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm8 = nn.BatchNorm2d(size_8)
        self.conv9 = nn.Conv2d(size_8, size_9, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm9 = nn.BatchNorm2d(size_9)
        self.shortcut4 = nn.Conv2d(size_7, size_9, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.normS4 = nn.BatchNorm2d(size_9)


        # BLOCK 3 #
        self.conv10 = nn.Conv2d(size_9, size_10, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.norm10 = nn.BatchNorm2d(size_10)
        self.conv11 = nn.Conv2d(size_10, size_11, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm11 = nn.BatchNorm2d(size_11)
        self.shortcut5 = nn.Conv2d(size_9, size_11, kernel_size = 1, stride = 2, padding = 0, bias = False)
        self.normS5 = nn.BatchNorm2d(size_11)
        
        self.conv12 = nn.Conv2d(size_11, size_12, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm12 = nn.BatchNorm2d(size_12)
        self.conv13 = nn.Conv2d(size_12, size_13, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm13 = nn.BatchNorm2d(size_13)
        self.shortcut6 = nn.Conv2d(size_11, size_13, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.normS6 = nn.BatchNorm2d(size_13)


        # BLOCK 4 #
        self.conv14 = nn.Conv2d(size_13, size_14, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.norm14 = nn.BatchNorm2d(size_14)
        self.conv15 = nn.Conv2d(size_14, size_15, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm15 = nn.BatchNorm2d(size_15)
        self.shortcut7 = nn.Conv2d(size_13, size_15, kernel_size = 1, stride = 2, padding = 0, bias = False)
        self.normS7 = nn.BatchNorm2d(size_15)
        
        self.conv16 = nn.Conv2d(size_15, size_16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm16 = nn.BatchNorm2d(size_16)
        self.conv17 = nn.Conv2d(size_16, size_17, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.norm17 = nn.BatchNorm2d(size_17)
        self.shortcut8 = nn.Conv2d(size_15, size_17, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.normS8 = nn.BatchNorm2d(size_17)

        self.linear = nn.Linear(512, num_classes)


    def forward(self, x0):
        x1 = F.relu(self.norm1(self.conv1(x0)))        # x1 has size 64 (i.e. it has 64 filters)

        # BLOCK 1 #
        x2 = F.relu(self.norm2(self.conv2(x1)))         # x2 has size 64
        x3 = F.relu(self.norm3(self.conv3(x2)))         # x3 has size 64
        xS1 = F.relu(self.normS1(self.shortcut1(x1)))   # have to project x1 to have the same size as x3
        x3 = x3 + xS1                                   
        x4 = F.relu(self.norm4(self.conv4(x3)))         # x4 has size 64
        x5 = F.relu(self.norm5(self.conv5(x4)))         # x5 has size 64
        xS2 = F.relu(self.normS2(self.shortcut2(x3)))   # have to project x3 to have the same size as x5
        x5 = x5 + xS2
        

        # BLOCK 2 #
        x6 = F.relu(self.norm6(self.conv6(x5)))         # x6 has size 128
        x7 = F.relu(self.norm7(self.conv7(x6)))         # x7 has size 128
        xS3 = F.relu(self.normS3(self.shortcut3(x5)))   # have to project x5 to have the same size as x7
        x7 = x7 + xS3
        x8 = F.relu(self.norm8(self.conv8(x7)))         # x8 has size 128
        x9 = F.relu(self.norm9(self.conv9(x8)))         # x9 has size 128
        xS4 = F.relu(self.normS4(self.shortcut4(x7)))   # have to project x7 to have the same size as x9
        x9 = x9 + xS4

        # BLOCK 3 #
        x10 = F.relu(self.norm10(self.conv10(x9)))      # x10 has size 256
        x11 = F.relu(self.norm11(self.conv11(x10)))     # x11 has size 256
        xS5 = F.relu(self.normS5(self.shortcut5(x9)))   # have to project x9 to have the same size as x11
        x11 = x11 + xS5
        x12 = F.relu(self.norm12(self.conv12(x11)))     # x12 has size 256
        x13 = F.relu(self.norm13(self.conv13(x12)))     # x13 has size 256
        xS6 = F.relu(self.normS6(self.shortcut6(x11)))  # have to project x11 to have the same size as x13
        x13 = x13 + xS6

        # BLOCK 4 #
        x14 = F.relu(self.norm14(self.conv14(x13)))     # x14 has size 512
        x15 = F.relu(self.norm15(self.conv15(x14)))     # x15 has size 512
        xS7 = F.relu(self.normS7(self.shortcut7(x13)))  # have to project x13 to have the same size as x15
        x15 = x15 + xS7
        x16 = F.relu(self.norm16(self.conv16(x15)))     # x16 has size 512
        x17 = F.relu(self.norm17(self.conv17(x16)))     # x17 has size 512
        xS8 = F.relu(self.normS8(self.shortcut8(x15)))  # have to project x15 to have the same size as x17
        x17 = x17 + xS8

        x18 = F.avg_pool2d(x17, 4)
        x18 = x18.view(x18.size(0), -1)
        x19 = self.linear(x18)

        output = x19
        activations = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17]

        return output

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3,
                                 64,
                                 kernel_size=7,
                                 stride=2,
                                 padding=3,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def build_model(model_name, num_classes, device, model_path, layer_idx):
    idx = []
    if model_name == "VGG19":
        model = VGG19(num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_name == "ResNet18":
        model = ResNet18(num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        for i in layer_idx:
            idx.append(resnet18[i])
    elif model_name == "ResNet18-pretrained-imagenet":
        model = ResNet(BasicBlock, [2, 2, 2, 2])
        pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.to(torch.device(device))
    else:
        raise ValueError("{} is not a valid model.".format(model_name))
    
    return model, idx
