#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    """ Highway layer implementation
    """
    def __init__(self, input_size, output_size):
        """ Init Highway layer.

        @param input_size (int): Input tensor size (dimensionality). This should be e_{word}.
        @param output_size (int): Output Size (dimensionality). This should also be e_{word}.
        """
        super(Highway, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.x_proj_layer = nn.Linear(input_size, output_size, bias=True)
        self.x_gate_layer = nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        """ Take a mini-batch of tensors and run it through the highway layer.

        @param x (Tensor): input tensor of shape (batch_size, input_size) 
        @returns x_highway (Tensor): output tensor of shape (batch_size, output_size)
        """
        x_proj = F.relu(self.x_proj_layer(x))
        x_gate = torch.sigmoid(self.x_gate_layer(x))
        x_highway = (x_gate * x_proj) + ((1-x_gate) * x)
        
        """ manual verification

        print(x_proj.shape)
        print(x_gate.shape)
        print(x_highway.shape)

        print(x_proj)
        print(x_gate)
        print(x_highway)
        """

        return x_highway

### END YOUR CODE 

