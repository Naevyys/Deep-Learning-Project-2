from src.module import Module
import torch
from torch import empty
from torch.nn.functional import fold, unfold


class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        """
        Initialize 2d convolution layer. We consider in our implementation that groups=1.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride of the kernel, default is 1.
        :param bias: Whether to include a bias term or not. Default True.
        """
        super().__init__()

        # Set layer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias

        # Initialize w
        self.w = torch.randn(size=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))

        # TODO:
        # - Add assertions to verify input types and shapes
        # - Accept kernel sizes & strides as tuples
        # - Add more parameters if needed (padding, dilatation, ...)
        # Note: Check whether in_channels argument is needed at all without the groups argument

    def __convolve(self, x):
        """
        Applies kernels to tensor x.
        :param x: Input tensor. Must be of size (batch_size, channels, height, width)
        :return: Output of the convolution.
        """

        # TODO:
        # - Implement basic convolution. Algorithm:
        #   - Unfold x
        #   - Flatten kernels
        #   - Multiply x and kernels appropriately
        #   - Fold back to output shape (what is the general formula for the output shape?)
        # - Multiplication for entire batch over all channels at once
        # - Account for bias
        # - Take stride into account
        # - Implement unit tests to validate implementation

        raise NotImplementedError

    def forward(self, *inputs):
        return self.__convolve(*inputs)

    def backward(self, *gradwrtoutput):
        raise NotImplementedError  # TODO

    def param(self):
        raise NotImplementedError  # TODO
