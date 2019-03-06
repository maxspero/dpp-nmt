#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """ Implementation of 1-dimensional CNN for character-based encoding
    """
    def __init__(self, in_channels, num_filters, kernel_size):
        """ Init CNN Layer.

        @param in_channels (int): Number of output channels.
                                  This should be e_{char}.
        @param num_filters (int): Number of output channels.
                                  This should be e_{word}.
        @param kernel_size (int): Number of elements in the conv window at any given time.
                                  This should be 5.
        """
        super(CNN, self).__init__()

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.conv1d = nn.Conv1d(in_channels, num_filters, kernel_size)

    def forward(self, x):
        """ Take a mini-batch of tensors and run it through the highway layer.

        @param x (Tensor): input tensor of shape (batch_size, in_channels, m_word) 
        @returns x_conv_out (Tensor): output tensor of shape (batch_size, num_filters)
        """
        x_conv = self.conv1d(x)
        x_conv_relu = F.relu(x_conv)
        x_conv_out, max_indices = torch.max(x_conv_relu, 2)
        
        """ manual verification
        print(x)
        print(x_conv)
        print(x_conv_relu)
        print(x_conv_out)

        print(x.shape)
        print(x_conv.shape)
        print(x_conv_relu.shape)
        print(x_conv_out.shape)
        """
        return x_conv_out

### END YOUR CODE 

