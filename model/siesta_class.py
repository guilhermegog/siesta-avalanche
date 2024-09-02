from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn.functional import F

from avalanche.models import DynamicModule


class SiestaClassifier(DynamicModule):
    def __init__(self, in_features, out_features, tau=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.register_buffer("class_counter", torch.zeros(
            out_features, dtype=torch.int64))

    def forward(self, x):
        norm_weights = F.normalize(self.weights, dim=1)
        norm_x = F.normalize(x, dim=1)

        cosine_similarity = norm_x @ norm_weights.t()
        scaled_cosine_sim = cosine_similarity/self.tau

        out = F.softmax(scaled_cosine_sim, dim=1)
        return out

    def online_update(self, x, y):
        self.weights[y.label] = self.class_counter[y.label] * \
            self.weights[y.label]+x
        self.weights[y.label] = self.weights[y.label] / \
            (self.class_counter[y.label]+1)

        self.class_counter[y.label] += 1
