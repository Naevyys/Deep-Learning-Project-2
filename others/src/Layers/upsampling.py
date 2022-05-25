from ...src.module import Module
from ...src.Layers.convolution import Conv2d
from ...src.Layers.nearest_neighbor_upsample import NNUpsampling


class Upsampling(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=1, bias=True,
                 transposeconvargs=True, scale_factor=None):
        """
        Initialize an upsampling block layer, which is composed of an Upsample layer followed by a Conv2d layer.
        The goal of this layer is to substitude a transpose convolution layer, so it will take arguments equivalent to
        the transpose convolution arguments, e.g. if you pass a stride, il will correspond to a stride value of a
        transpose convolution layer. This means that the output shape of the Upsampling layer is expected to be the same
        as the output shape of a transpose convolution with the same arguments (with output_padding = stride - 1).
        Note: If you do not keep the dilation argument to 1, the output is not guaranteed to have the same shape as for
        a transpose convolution, as we were unable to correctly figure out the equivalence for this parameter.
        This is the implemented equivalence of transpose convolution arguments to NNUpsample2d and Conv2d arguments
        ( > layer argument == mapping of Upsampling argument):
        [NNUpsample2d]
          > scale_factor == stride
        [Conv2d]
          > in_channels == in_channels
          > out_channels == out_channels
          > kernel_size == kernel_size
          > stride == dilation
          > dilation == dilation
          > padding == dilation * (kernel_size - 1) - padding
          > bias == bias
        Note that the kernel_size, dilation and padding must satisfy dilation*(kernel_size - 1) - padding >= 1 .
        Please set 'transposeconvargs' to False and pass a factor > 0 if you want the arguments passed to correspond to
        the standard convolution arguments of the Conv2d following the NNUpsample2d.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride value (int or tuple of int)
        :param dilation: Dilation value (int or tuple of ints)
        :param padding: Padding value (int or tuple of ints).
        :param bias: True if we want a bias.
        :param transposeconvargs: True if arguments passed correspond to transpose convolution arguments.
        :param scale_factor: Upsampling factor for Upsample layer, required if transposeconvargs is False.
        """

        if transposeconvargs:
            assert dilation*(kernel_size - 1) - padding >= 1, "You do not satisfy dilation*(kernel_size - 1) - padding >= 1, this results in negative padding!"
            # Other assertions are already done in the respective layers.

        factor_ = stride if transposeconvargs else scale_factor
        stride_ = dilation if transposeconvargs else stride
        padding_ = dilation * (kernel_size - 1) - padding if transposeconvargs else padding
        dilation_ = 1 if transposeconvargs else dilation

        self.upsample = NNUpsampling(factor_)
        self.conv2d = Conv2d(in_channels, out_channels, kernel_size, stride=stride_, dilation=dilation_, padding=padding_, bias=bias)

    def forward(self, *inputs):
        return self.conv2d(self.upsample(*inputs))

    def backward(self, *gradwrtoutput):
        return self.upsample.backward(self.conv2d.backward(*gradwrtoutput))

    def param(self):
        return self.conv2d.param()

    def update_param(self, updated_params): 
        self.conv2d.update_param(updated_params)

    def zero_grad(self):
        self.conv2d.zero_grad()