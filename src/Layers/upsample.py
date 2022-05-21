from src.module import Module
from torch import empty, cat, arange
from src.utils import conv2d


class Upsample2d(Module):

    def __init__(self, factor):
        """
        Initialize a nearest neighbor upsample layer
        :param factor: Upsampling factor
        """

        self.factor = factor
        self.in_shape = None

    def forward(self, *inputs):
        """
        Upsample the inputs (nearest neighbor upsampling)
        :param inputs: Tensor of size (batch_size, channels, height, width)
        :return: Upsampled input
        """

        x = inputs[0]
        self.n_channels = x.shape[-3]

        out_shape = list(x.shape)  # (batch_size, channels, height, width) or (channels, height, width)
        out_shape[-2], out_shape[-1] = out_shape[-2] * self.factor, out_shape[-1] * self.factor
        x = x.unsqueeze(-2).unsqueeze(-1)
        x = x.tile((1, 1, 1, self.factor, 1, self.factor))  # Also works without batch dimension
        x = x.reshape(out_shape)  # (batch_size, channels, height*factor, width*factor) or same without batch_size
        return x

    def backward(self, *gradwrtoutput):
        # There are no parameters to update in the backward pass of the upsampling layer. We only need to downsample the
        # gradient wrt output to match the shape of the input to this layer by summing the gradients of outputs
        # corresponding to the same input entry.
        # This sum can be implemented as a convolution with a specially crafted filter and parameters.

        dl_ds = gradwrtoutput[0]

        kernel_sum = empty(size=(self.n_channels, self.n_channels, self.factor, self.factor)).double().zero_()
        for i in range(self.n_channels):
            kernel_sum[i, i, :, :] = 1
        dl_dx_previous_layer = conv2d(dl_ds, kernel_sum, stride=self.factor)

        return dl_dx_previous_layer
