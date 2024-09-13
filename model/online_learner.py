import torch
import torch.nn as nn


class RunningMean(nn.Module):

    def __init__(self, input_shape, num_classes, device='cuda:2'):

        super(RunningMean, self).__init__()

        # parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes

        # initialize
        self.muK = torch.zeros((num_classes, input_shape),
                               dtype=torch.float64).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)

    @torch.no_grad()
    def fit(self, x, y):
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        # update class means
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
        self.cK[y] += 1

    def grab_mean(self, y):
        return self.muK[y]
