import math

import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init


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
            self.register_parameter("sigma", None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features, 1))
        else:
            self.register_parameter("bias", None)
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
            input = torch.cat((input, (torch.ones(len(input), 1).cuda("cuda:2"))), dim=1)
            concat_weight = torch.cat((self.weight, self.bias), dim=1)
            out = F.linear(
                F.normalize(input, p=2, dim=1, eps=1e-8),
                F.normalize(concat_weight, p=2, dim=1, eps=1e-8),
            )
        else:
            out = F.linear(
                F.normalize(input, p=2, dim=1, eps=1e-8),
                F.normalize(self.weight, p=2, dim=1, eps=1e-8),
            )

        if self.sigma is not None:
            out = self.sigma * out
        return out
