import torch
import copy
import numpy as np
import torch.nn as nn
from einops import reduce, rearrange
from timm.models.layers import DropPath, trunc_normal_


class conv_bn(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, groups=1, is_bn=True):
        super().__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels) if is_bn else nn.Identity()
        )


class repBlock(nn.Module):
    def __init__(self, in_channels, drop_path=0, deploy=False, expansion=4, layer_scale_init_value=1e-6):
        super().__init__()
        self.deploy = deploy
        self.groups = in_channels
        self.in_channels = in_channels

        self.nonlinearity = nn.ReLU(inplace=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        if deploy:
            self.rep_conv = nn.Conv2d(in_channels, in_channels, 7, 1, 3, groups=in_channels)
            self.rep_1x1_in = nn.Conv2d(in_channels, int(in_channels * expansion), 1, 1)
            self.rep_1x1_out = nn.Conv2d(int(in_channels * expansion), in_channels, 1, 1)
        else:
            self.dw3x3 = conv_bn(in_channels, in_channels, 3, 1, 1, 1, in_channels)
            self.dw5x5 = conv_bn(in_channels, in_channels, 5, 1, 2, 1, in_channels)
            self.dw7x7 = conv_bn(in_channels, in_channels, 7, 1, 3, 1, in_channels)
            self.conv1x1_in = conv_bn(in_channels, int(in_channels * expansion), 1, 1, 0)
            self.conv1x1_out = conv_bn(int(in_channels * expansion), in_channels, 1, 1, 0)

    def forward(self, x):
        inp = x
        if hasattr(self, 'rep_conv'):
            x = self.rep_conv(x)
            x = self.rep_1x1_in(x)
            x = self.nonlinearity(x)
            x = self.rep_1x1_out(x)
        else:
            x = self.dw3x3(x) + self.dw5x5(x) + self.dw7x7(x)
            x = self.conv1x1_in(x)
            x = self.nonlinearity(x)
            x = self.conv1x1_out(x)

        if self.gamma is not None:
            x = rearrange(x, 'B C H W -> B H W C')
            x = self.gamma * x
            x = rearrange(x, 'B H W C -> B C H W')
        x = inp + self.drop_path(x)

        return x

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.dw3x3)
        kernel5x5, bias5x5 = self._fuse_bn_tensor(self.dw5x5)
        kernel7x7, bias7x7 = self._fuse_bn_tensor(self.dw7x7)

        rep_conv_kernel = kernel7x7 + self._pad_tensor(kernel5x5, 1) + self._pad_tensor(kernel3x3, 2)
        rep_conv_bias = bias7x7 + bias5x5 + bias3x3

        kernel_1x1_in, kernel_1x1_in_bias = self._fuse_bn_tensor(self.conv1x1_in)
        kernel_1x1_out, kernel_1x1_out_bias = self._fuse_bn_tensor(self.conv1x1_out)

        return rep_conv_kernel, rep_conv_bias, kernel_1x1_in, kernel_1x1_in_bias, kernel_1x1_out, kernel_1x1_out_bias

    def _pad_tensor(self, kernel, padding):
        return torch.nn.functional.pad(kernel, [padding, padding, padding, padding])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rep_conv'):
            return
        rep_conv_kernel, rep_conv_bias, kernel_1x1_in, kernel_1x1_in_bias, kernel_1x1_out, kernel_1x1_out_bias = self.get_equivalent_kernel_bias()
        self.rep_conv = nn.Conv2d(in_channels=self.dw7x7[0].in_channels, out_channels=self.dw7x7[0].out_channels,
                                  kernel_size=self.dw7x7[0].kernel_size, stride=self.dw7x7[0].stride,
                                  padding=self.dw7x7[0].padding, dilation=self.dw7x7[0].dilation,
                                  groups=self.dw7x7[0].groups, bias=True)
        self.rep_conv.weight.data = rep_conv_kernel
        self.rep_conv.bias.data = rep_conv_bias

        self.rep_1x1_in = nn.Conv2d(self.conv1x1_in[0].in_channels, self.conv1x1_in[0].out_channels, 1, 1)
        self.rep_1x1_in.weight = nn.Parameter(kernel_1x1_in)
        self.rep_1x1_in.bias = nn.Parameter(kernel_1x1_in_bias)

        self.rep_1x1_out = nn.Conv2d(self.conv1x1_out[0].in_channels, self.conv1x1_out[0].out_channels, 1, 1)
        self.rep_1x1_out.weight = nn.Parameter(kernel_1x1_out)
        self.rep_1x1_out.bias = nn.Parameter(kernel_1x1_out_bias)

        self.__delattr__('dw7x7')
        self.__delattr__('dw5x5')
        self.__delattr__('dw3x3')
        self.__delattr__('conv1x1_in')
        self.__delattr__('conv1x1_out')

        self.deploy = True


class downsample(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.deploy = deploy

        if self.deploy:
            self.rep_conv = nn.Conv2d(in_channels, out_channels, 2, 2)
        else:
            self.conv = conv_bn(in_channels, out_channels, 2, 2, 0)

    def forward(self, x):
        if hasattr(self, 'rep_conv'):
            return self.rep_conv(x)

        x = self.conv(x)

        return x

    def get_equivalent_kernel_bias(self):
        kernel2x2, bias2x2 = self._fuse_bn_tensor(self.conv)

        return kernel2x2, bias2x2

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rep_conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rep_conv = nn.Conv2d(in_channels=self.conv[0].in_channels, out_channels=self.conv[0].out_channels,
                                  kernel_size=self.conv[0].kernel_size, stride=self.conv[0].stride,
                                  padding=self.conv[0].padding, dilation=self.conv[0].dilation,
                                  groups=self.conv[0].groups, bias=True)
        self.rep_conv.weight.data = kernel
        self.rep_conv.bias.data = bias
        self.__delattr__('conv')

        self.deploy = True


class stem(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.deploy = deploy

        if self.deploy:
            self.rep_conv = nn.Conv2d(in_channels, out_channels, 4, 4)
        else:
            self.conv = conv_bn(in_channels, out_channels, 4, 4, 0)

    def forward(self, x):
        if hasattr(self, 'rep_conv'):
            return self.rep_conv(x)
        x = self.conv(x)

        return x

    def get_equivalent_kernel_bias(self):
        kernel2x2, bias2x2 = self._fuse_bn_tensor(self.conv)

        return kernel2x2, bias2x2

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rep_conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rep_conv = nn.Conv2d(in_channels=self.conv[0].in_channels, out_channels=self.conv[0].out_channels,
                                  kernel_size=self.conv[0].kernel_size, stride=self.conv[0].stride,
                                  padding=self.conv[0].padding, dilation=self.conv[0].dilation,
                                  groups=self.conv[0].groups, bias=True)
        self.rep_conv.weight.data = kernel
        self.rep_conv.bias.data = bias
        self.__delattr__('conv')

        self.deploy = True


class RepMNet(nn.Module):
    def __init__(self, num_classes=102, drop_path_rate=0., dim=[96, 192, 384, 768], deeps=[3, 3, 9, 3],
                 layer_scale_init_value=1e-6, head_init_scale=1., deploy=False):
        super().__init__()

        self.deploy = deploy

        self.stem = stem(3, dim[0], deploy=self.deploy)

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(deeps))]

        self.stage_1 = self._make_stage(dim[0], dim[1], deeps[0], dp_rates[:3], layer_scale_init_value)
        self.stage_2 = self._make_stage(dim[1], dim[2], deeps[1], dp_rates[3:6], layer_scale_init_value)
        self.stage_3 = self._make_stage(dim[2], dim[3], deeps[2], dp_rates[6:15], layer_scale_init_value)
        self.stage_4 = self._make_stage(dim[3], dim[3], deeps[3], dp_rates[15:18], layer_scale_init_value,
                                        is_downsample=False)

        self.fc = nn.Linear(dim[-1], num_classes)

        self.apply(self._init_weights)
        self.fc.weight.data.mul_(head_init_scale)
        self.fc.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)

        return reduce(x, 'B C H W -> B C', 'mean')

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)

        return x

    def _make_stage(self, in_channels, out_channels, deep, dp_rates, layer_scale_init_value, is_downsample=True):
        blocks = []

        for i in range(deep):
            blocks.append(
                repBlock(in_channels, dp_rates[i], layer_scale_init_value=layer_scale_init_value, deploy=self.deploy))

        if is_downsample:
            blocks.append(downsample(in_channels, out_channels, deploy=self.deploy))

        return nn.Sequential(*blocks)


def model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


if __name__ == '__main__':
    from thop import profile

    net = RepMNet(drop_path_rate=0.1)

    net.eval()
    checkpoint = torch.load('../exp8-72.0%/best.pth', map_location='cpu')
    msg = net.load_state_dict(checkpoint['model'])
    print(msg)
    rep_net = model_convert(net, '../exp8-72.0%/rep_best.pth')
    # net = RepMNet()
    # print(net)
    # msg = net.load_state_dict(torch.load('../exp8-72.0%/rep_best.pth', map_location='cpu'))
    # print(msg)