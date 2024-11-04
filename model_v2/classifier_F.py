import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import init
import math
from torch.nn.parameter import Parameter


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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


class Classifier(nn.Module):
    def __init__(self, init_features, embed_size, num_classes, distance_measure="CosineLinear"):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2048, init_features),
            nn.ReLU(),
            nn.Linear(init_features, embed_size),
            nn.ReLU(),
            #nn.Dropout(p=0.2, inplace=True),
            CosineLinear(embed_size, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)

    def _get_penultimate_feature(self, x):
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        return self.classifier[2](x)
