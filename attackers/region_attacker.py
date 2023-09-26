from pickletools import optimize
from turtle import forward
from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from torch import Tensor

from typing import  Callable, Optional
import timm 

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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


class DummyAttacker(nn.Module):
    """_summary_
    v1.0 -> 0.258  |  0.214   |  0.327  | 0.250  |  0.164   |  0.328 
    Args:
        nn (_type_): _description_
    """
    def __init__(self, c_in=3,  c_hid=512, c_out=1) -> None:
        super().__init__()
        self.norm1 = LayerNorm(c_in,eps=1e-6,data_format="channels_first")
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_hid, 1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_hid, c_hid, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        # self.conv2 = timm.create_model('resnet18',features_only=True, output_stride=8, pretrained=True)

        # self.conv2 = BasicBlock(inplanes=c_hid, planes=c_hid)
        self.conv3 = nn.Conv2d(c_hid, c_out, 1)
    def forward(self, x):
        x = self.norm1(x)
        x = self.conv1(x)
        # x = self.conv2(x)[-1]
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class RegionAttacker(nn.Module):
    def __init__(self, c_in=3,  c_hid=64, c_out=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, c_hid, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(c_hid)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blk = BasicBlock(inplanes=c_hid, planes=c_hid)

        self.out = nn.Conv2d(c_hid, c_out, 3, padding=1)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 1/4 downsampling
        x = self.blk(x)

        y = self.out(x)

        return y



def ground_truth_generator(loss):
    gt = torch.zeros_like(loss)
    gt = torch.where(loss > -torch.log(torch.tensor(0.5)), torch.ones_like(loss), gt )
    return gt


def train_attacker(grad, loss):
    """_summary_

    Args:
        grad (_type_): _description_
        loss (_type_): _description_
    """
    model = RegionAttacker()
    optimizer = torch.optim.Adam( params=model.parameters(),  lr=0.1, weight_decay=0.01)
    # imgae gradient as input, output high value
    # ground truth is segmentation loss > - log 0.5
    y = model(grad)

    gt = ground_truth_generator(loss)

    loss = F.binary_cross_entropy(y, gt)
    loss.backward()
    optimizer.step()

    return model

if __name__ == "__main__":
    attcker = RegionAttacker()

    x = torch.rand(1, 3, 480, 854)

    y = attcker(x)
    print(y.shape)