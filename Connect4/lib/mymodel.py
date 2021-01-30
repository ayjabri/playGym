#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 08:06:01 2021

@author: Ayman Jabri

Residual Neural Network to play connect4 game
"""

from numpy import prod

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, obs_shape=(6, 7), players=2, history=4, filters=64):
        super().__init__()

        self.obs_shape = obs_shape
        self.history = history
        self.filters = filters
        self.input_shape = history * players

        self.input = nn.Sequential(nn.Conv2d(self.input_shape, filters, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(filters),
                                   nn.ReLU(inplace=True),
                                   )

        self.layer1 = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(filters),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(filters, filters,
                                              kernel_size=3, padding=1),
                                    nn.BatchNorm2d(filters),
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(filters),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(filters, filters,
                                              kernel_size=3, padding=1),
                                    nn.BatchNorm2d(filters),
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(filters),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(filters, filters,
                                              kernel_size=3, padding=1),
                                    nn.BatchNorm2d(filters),
                                    )

        fc_input_size = self._conv_output_shape()

        # Value represents the probability of winning [0,1]
        self.value = nn.Sequential(nn.Linear(fc_input_size, 256),
                                   nn.LeakyReLU(),
                                   nn.Linear(256, 1),
                                   nn.Sigmoid())

        self.policy = nn.Sequential(nn.Linear(fc_input_size, 256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, self.obs_shape[1]))

    def _forward(self, x):
        y = self.input(x)
        output = y + self.layer1(y)
        F.leaky_relu_(output)
        output = y + self.layer2(y)
        F.leaky_relu_(output)
        output = y + self.layer3(y)
        return F.leaky_relu_(output)

    def _conv_output_shape(self):
        o = torch.zeros(1, self.input_shape, *self.obs_shape)
        output_shape = list(self._forward(o).shape)[1:]
        return prod(output_shape)

    def forward(self, x):
        output = self._forward(x)
        output = torch.flatten(output, start_dim=1, end_dim=-1)
        value = self.value(output)
        policy = self.policy(output)
        return policy, value
