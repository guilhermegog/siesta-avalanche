import torch
import torch.nn as nn

from model.conv_net_pt import Net
from model.siesta_class import SiestaClassifier
from model.siesta_g_net import GNet


class SiestaNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        pretrained_layers = torch.load("model/conv_layers.pth")
        conv_pt = Net()
        conv_pt.load_state_dict(pretrained_layers, strict=False)
        conv_pt.eval()

        self.h_net = conv_pt
        self.g_layers = GNet(512 * 2 * 2, 1024)

        self.f_classifier = SiestaClassifier(
            self.g_layers.get_output_features(), num_classes
        )

        self.register_buffer("stored_samples_nr", torch.zeros(num_classes))

    def forward(self, x, sleep=False):

        x = self.h_net.bn1(self.h_net.conv1(x))
        x = torch.relu(x)

        x = self.h_net.bn2(self.h_net.conv2(x))
        x = torch.relu(x)

        x = self.h_net.bn3(self.h_net.conv3(x))
        x = torch.relu(x)

        x = self.h_net.bn4(self.h_net.conv4(x))
        x = torch.relu(x)

        x = self.h_net.bn5(self.h_net.conv5(x))
        lr = torch.relu(x)

        lr = lr.view(-1, 512 * 2 * 2)
        x = self.g_layers(lr)
        #x = x.squeeze()
        out, z = self.f_classifier(x, sleep)
        if sleep:
            return out
        else:
            return out, z.squeeze(), lr
