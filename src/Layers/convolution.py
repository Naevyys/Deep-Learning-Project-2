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
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        self.bias = bias

        # Initialize w
        self.w = torch.randn(size=(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])).double()

        # TODO:
        # - Add assertions to verify input types and shapes
        # - Accept kernel sizes & strides as tuples
        # - Add more parameters if needed (padding, dilatation, ...)

    def __convolve(self, x):
        """
        Applies kernels to tensor x.
        :param x: Input tensor. Must be of size (batch_size, channels, height, width)
        :return: Output of the convolution.
        """

        batch_size, _, height, width = x.shape

        unfolded = unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        wxb = self.w.view(self.out_channels, -1) @ unfolded + 0  # TODO: add bias
        result = wxb.view(batch_size, self.out_channels, height - self.kernel_size[0] + 1, width - self.kernel_size[1] + 1)

        return result

        # TODO:
        # - Account for bias
        # - Take stride into account
        # - Implement unit tests to validate implementation

    def forward(self, *inputs):
        return self.__convolve(*inputs)

    def backward(self, *gradwrtoutput):
        raise NotImplementedError  # TODO

    def param(self):
        raise NotImplementedError  # TODO
