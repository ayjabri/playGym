# My Conv models

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DQN(nn.Module):
    '''
    Basic Deep Q-Learning Network with 3 Conv layers and two fully connected
    output layers
    '''
    def __init__(self, f_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(f_shape[0], 32, 8, 4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 1),
                                  nn.ReLU(),
                                  nn.Flatten()
                                  )
        outshape = self.conv(torch.zeros(1, *f_shape)).shape[1]
        self.fc = nn.Sequential(nn.Linear(outshape, 512),
                                nn.ReLU(),
                                nn.Linear(512, n_actions))

    def forward(self, x):
        x = self.conv(x.float()/256)
        return self.fc(x)



class DuelDQN(nn.Module):
    '''
    Duel Deep Q-Learning Networ with 3 Convolutional layers and duel fully connected
    layers
    '''
    def __init__(self, f_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(f_shape[0], 32, 8, 4),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(32),
                                  nn.Conv2d(32, 64, 4, 2),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(64),
                                  nn.Conv2d(64, 64, 3, 1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(64),
                                  nn.Flatten()
                                  )
        outshape = self.conv(torch.zeros(1, *f_shape)).shape[1]
        self.fc_adv = nn.Sequential(nn.Linear(outshape, 256),
                                nn.ReLU(),
                                nn.Linear(256, n_actions))
        self.fc_val = nn.Sequential(nn.Linear(outshape, 256),
                                nn.ReLU(),
                                nn.Linear(256, 1))

    def forward(self, x):
        x = self.conv(x.float()/256)
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        return (val + (adv - adv.mean(dim=1, keepdim=True)))


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        if bias:
            b = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(b)
            self.register_buffer('epsilon_bias', torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.data.normal_()
        w = self.weight + self.sigma_weight * self.epsilon_weight.data
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.data.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, w, bias)



class NoisyDQN(nn.Module):
    def __init__(self, f_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(f_shape[0], 32, 8, 4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 1),
                                  nn.ReLU(),
                                  nn.Flatten()
                                  )
        outshape = self.conv(torch.zeros(1, *f_shape)).shape[1]

        self.fc = nn.Sequential(NoisyLinear(outshape, 512),
                                nn.ReLU(),
                                NoisyLinear(512, n_actions))

    def forward(self, x):
        x = self.conv(x.float() / 256)
        return self.fc(x)



class DuelNoisyDQN(nn.Module):
    def __init__(self, f_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(f_shape[0], 32, 8, 4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 1),
                                  nn.ReLU(),
                                  nn.Flatten()
                                  )
        outshape = self.conv(torch.zeros(1, *f_shape)).shape[1]

        self.fc_adv = nn.Sequential(NoisyLinear(outshape, 256),
                                nn.ReLU(),
                                NoisyLinear(256, n_actions))
        self.fc_val = nn.Sequential(NoisyLinear(outshape, 256),
                                nn.ReLU(),
                                NoisyLinear(256, 1))

    def forward(self, x):
        x = self.conv(x.float() / 256)
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        return (val + (adv - adv.mean(dim=1, keepdim=True)))

