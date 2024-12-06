import torch
import torch.nn as nn

from torchvision.models import efficientnet_b0

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
    def __init__(self, latent_layer=8, num_classes=None, arch='mobilenet_v3_large'):
        super(MobNet_ClassifierF, self).__init__()
        self.model = efficientnet_b0(weights='IMAGENET1K_V1')
        self.num_layer = 17
        self.num_stages = 8
        assert latent_layer < self.num_layer, "Network split invalid."

        stage_layer_dict = {'0': [0],
                            '1': [1],
                            '2': [2, 3],
                            '3': [4, 5],
                            '4': [6, 7, 8],
                            '5': [9, 10, 11],
                            '6': [11, 12, 13, 14],
                            '7': [15],
                            '8': [16]}

        for stage, layers in stage_layer_dict.items():
            if(latent_layer not in layers):
                del self.model.features[0]
            else:
                i = 0
                for name, module in self.model.features[0].named_modules():
                    curr_layer = layers[i]
                    i = i+1
                    if(curr_layer < latent_layer):
                        del self.model.features[0][0]

                    else:
                        break
                break

        print(self.model.features)
        if num_classes is not None:
            print('Changing output layer to contain %d classes. ' % num_classes)
            self.model.classifier[-1] = CosineLinear(
                1280, num_classes)

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
    def __init__(self, latent_layer=8, arch='mobilenet_v3_large'):
        super(MobNet_ClassifierG, self).__init__()
        self.model = efficientnet_b0(weights='IMAGENET1K_V1')
        self.num_layer = 17
        self.num_stages = 8
        assert latent_layer < self.num_layer, "Network split invalid."

        stage_layer_dict = {'0': [0],
                            '1': [1],
                            '2': [2, 3],
                            '3': [4, 5],
                            '4': [6, 7, 8],
                            '5': [9, 10, 11],
                            '6': [11, 12, 13, 14],
                            '7': [15],
                            '8': [16]}

        for stage in range(8, -1, -1):
            if(latent_layer not in stage_layer_dict[str(stage)]):
                del self.model.features[-1]
            else:
                layers = stage_layer_dict[str(stage)]
                i = len(layers)
                for name, module in self.model.features[-1].named_modules():
                    curr_layer = layers[i-1]
                    i -= 1
                    if(curr_layer > latent_layer-1):
                        del self.model.features[-1][-1]
                    else:
                        break
                break

        del self.model.classifier

    def forward(self, x):
        out = self.model.features(x)
        return out
