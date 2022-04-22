from src.module import Module
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold


class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        """
        Initialize 2d convolution layer. We consider in our implementation that groups=1.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride of the kernel, default is 1.
        :param dilation: Dilation of the kernel, default is 1.
        :param bias: Whether to include a bias term or not. Default True.
        """

        # Verify input types and shapes
        assert isinstance(in_channels, int), "in_channels must be an integer!"
        assert isinstance(out_channels, int), "out_channels must be an integer!"
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), "kernel_size must be an int or a tuple!"
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "kernel_size tuple must be of length 2!"
            assert isinstance(kernel_size[0], int) and isinstance(kernel_size[1], int), "kernel_size tuple value must be ints!"
        assert isinstance(stride, int) or isinstance(stride, tuple), "stride must be an int or a tuple!"
        if isinstance(stride, tuple):
            assert len(stride) == 2, "stride tuple must be of length 2!"
            assert isinstance(stride[0], int) and isinstance(stride[1], int), "stride tuple value must be ints!"
        assert isinstance(dilation, int) or isinstance(dilation, tuple), "dilation must be an int or a tuple!"
        if isinstance(dilation, tuple):
            assert len(dilation) == 2, "dilation tuple must be of length 2!"
            assert isinstance(dilation[0], int) and isinstance(dilation[1], int), "dilation tuple value must be ints!"
        assert isinstance(bias, bool), "bias must be a boolean!"

        super().__init__()

        # Set layer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        # Initialize w and bias
        self.w = empty(size=(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])).double().random_() / 2**53
        if bias:
            self.bias = empty(size=(self.out_channels,)).double().random_() / 2**53
        else:
            self.bias = empty(size=(self.out_channels,)).double().zero_()

        # Initialize parameters that do not have values yet.
        self.x_previous_layer = None
        self.dl_dw = empty(size=self.w.size())
        self.dl_db = empty(size=self.bias.size())

        # TODO:
        # - Add initialisation parameter to regulate initialisation of weights and bias to avoid vanishing gradient
        #   (waiting for answer to question to TAs)

    def __convolve(self, x):
        """
        Applies kernels to tensor x.
        :param x: Input tensor. Must be of size (batch_size, channels, height, width)
        :return: Output of the convolution.
        """

        self.x_previous_layer = x  # Store output of previous layer during forward pass, to be used in the backward pass

        batch_size, _, height, width = x.shape

        unfolded = unfold(x, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation)
        wxb = self.w.view(self.out_channels, -1) @ unfolded + self.bias.view(1, -1, 1)
        result = wxb.view(batch_size, self.out_channels, (height - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1,
                          (width - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1)

        return result

    def forward(self, *inputs):
        return self.__convolve(*inputs)

    def backward(self, *gradwrtoutput):
        # Note: the update is performed by the optimizer

        dl_ds = gradwrtoutput[0]

        if self.x_previous_layer is not None:  # No gradient if we don't have the output from the previous layer
            self.dl_dw = dl_ds.view(-1, 1).mm(self.x_previous_layer.view(1, -1))
            self.dl_db = dl_ds

        dl_dx_previous_layer = self.w.t().mv(dl_ds)

        return dl_dx_previous_layer

    def param(self):
        raise NotImplementedError  # TODO
