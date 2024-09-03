from typing import Dict

import torch
import torch.nn.functional as F
from avalanche.models import DynamicModule
from torch import Tensor, nn


class SiestaClassifier(DynamicModule):
    def __init__(self, in_features, out_features, tau=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.register_buffer(
            "class_counter", torch.zeros(out_features, dtype=torch.int64)
        )

    def forward(self, x):
        z = x
        norm_weights = F.normalize(self.weights, p=2, dim=1)
        norm_x = F.normalize(x, p=2, dim=1)

        cosine_similarity = torch.matmul(norm_x, norm_weights)
        scaled_cosine_sim = cosine_similarity / self.tau

        out = F.softmax(scaled_cosine_sim, dim=1)
        return out, z

    def online_update(self, x, label):
        current_class = self.class_counter[label]
        class_weights = self.weights[:, label]

        print(current_class)
        print(class_weights.shape)
        class_weights = torch.mul(current_class.item(), class_weights) + x
        class_weights = torch.div(
            class_weights, (current_class.item() + 1)
        )
        with torch.no_grad():
            self.weights[:, label] = class_weights

        self.class_counter[label] += 1
