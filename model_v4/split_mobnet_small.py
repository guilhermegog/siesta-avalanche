import torch
import torch.nn as nn

from torchvision.models import mobilenet_v3_small

from torch.nn import Parameter
import math
from torch.nn import functional as F
from torch.nn import init
from typing import Any, Callable, Dict, Optional


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # C x d i.e., 1000 x 1280
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0, 0.01)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is not None:
            input = torch.cat(
                (input, (torch.ones(len(input), 1).cuda())), dim=1)
            concat_weight = torch.cat((self.weight, self.bias), dim=1)
            out = F.linear(F.normalize(input, p=2, dim=1, eps=1e-8),
                           F.normalize(concat_weight, p=2, dim=1, eps=1e-8))
        else:
            out = F.linear(F.normalize(input, p=2, dim=1, eps=1e-8),
                           F.normalize(self.weight, p=2, dim=1, eps=1e-8))

        if self.sigma is not None:
            out = self.sigma * out
        return out


class MobNet_ClassifierF(nn.Module):
    def __init__(self, latent_layer=8, num_classes=None):
        super(MobNet_ClassifierF, self).__init__()

        self.model = mobilenet_v3_small(weights='IMAGENET1K_V1')

        for _ in range(0, latent_layer):
            del self.model.features[0]

        if num_classes is not None:
            print('Changing output layer to contain %d classes. ' % num_classes)
            self.model.classifier[3] = CosineLinear(1280, num_classes)

        print(self.model.features)

    def forward(self, x, feat=False):
        out = self.model.features(x)
        out = self.model.avgpool(out)
        out = torch.flatten(out, 1)

        if feat:
            features = self.model.classifier[0](out)
            out = self.model.classifier(out)
            return features, out

        out = self.model.classifier(out)
        return out


class MobNet_ClassifierG(nn.Module):
    def __init__(self, latent_layer=8):
        super(MobNet_ClassifierG, self).__init__()

        self.model = mobilenet_v3_small(weights='IMAGENET1K_V1')

        for _ in range(0, latent_layer+1):
            del self.model.features[-1]

        print(self.model.features)

    def forward(self, x, feat=False):
        out = self.model.features(x)
        out = self.model.avgpool(out)
        out = torch.flatten(out, 1)

        if feat:
            features = self.model.classifier[0](out)
            out = self.model.classifier(out)
            return features, out

        out = self.model.classifier(out)
        return out
