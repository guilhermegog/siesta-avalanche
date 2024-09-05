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
        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        self.register_buffer(
            "class_counter", torch.zeros(out_features, dtype=torch.int64)
        )

    def forward(self, x, sleep=False):
        z = x

        # Ensure input is of shape [B x N_feat x 1]
        if sleep:
            z = z.transpose(1, 2)
        else:
            z = x.view(1, x.size(1), x.size(0))

        norm_z = torch.linalg.vector_norm(z, ord=2)
        norm_weights = torch.linalg.vector_norm(self.weights, ord=2, dim=0)
        norm_factor = torch.mul(norm_weights, norm_z.squeeze())

        a_k = torch.matmul(self.weights.t(), z)

        # Shapes are of type [Batch, N_Class, 1]

        # Make sure [B, N, 1] / [1,N,1]
        norm_factor = norm_factor.view(1, self.out_features, 1)
        a_k = torch.div(a_k, norm_factor)
        logits = torch.div(a_k, self.tau)

        # [B x N] for softmax
        logits = logits.squeeze(-1)

        prob = F.softmax(logits, dim=1)
        out = torch.log(prob)
        if sleep:
            return out
        else:
            return out, z

    def online_update(self, x, label):
        current_class = self.class_counter[label]
        class_weights = self.weights[:, label]

        class_weights = torch.mul(current_class.item(), class_weights) + x
        class_weights = torch.div(class_weights, (current_class.item() + 1))

        with torch.no_grad():
            self.weights[:, label] = class_weights

        self.class_counter[label] += 1
