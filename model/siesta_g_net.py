import torch
import torch.nn as nn


class GNet(nn.Module):
    def __init__(self, input_features, output_features):
        super(GNet, self).__init__()
        self.gnet = nn.Sequential(
            nn.Linear(input_features, input_features),
            nn.ReLU(),
            nn.Linear(input_features, input_features),
            nn.ReLU(),
            nn.Linear(input_features, output_features),
            
        )

    def forward(self, x):
        out = self.gnet(x)
        out = torch.relu(out)
        return out

    @torch.no_grad()
    def get_output_features(self):
        return self.gnet[-1].out_features
