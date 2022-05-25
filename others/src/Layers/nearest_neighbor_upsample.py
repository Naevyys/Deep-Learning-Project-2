from ...src.module import Module
from torch import empty 
from ...src.utils import conv2d


class NNUpsample(Module):

    def __init__(self, scale_factor):
        """
        Initialize a nearest neighbor upsample layer
        :param factor: Upsampling factor, positive int
        """

        assert isinstance(scale_factor, int), "Factor must be an integer!"
        assert scale_factor > 0, "Factor must be positive!"

        self.scale_factor = scale_factor
        self.n_channels = None

    def forward(self, *inputs):
        """
        Upsample the inputs (nearest neighbor upsampling)
        :param inputs: Tensor of size (batch_size, channels, height, width)
        :return: Upsampled input
        """

        x = inputs[0]
        self.n_channels = x.shape[-3]

        out_shape = list(x.shape)  # (batch_size, channels, height, width) or (channels, height, width)
        out_shape[-2], out_shape[-1] = out_shape[-2] * self.scale_factor, out_shape[-1] * self.scale_factor
        x = x.unsqueeze(-2).unsqueeze(-1)
        x = x.tile((1, 1, 1, self.scale_factor, 1, self.scale_factor))  # Also works without batch dimension
        x = x.reshape(out_shape)  # (batch_size, channels, height*factor, width*factor) or same without batch_size
        return x

    def backward(self, *gradwrtoutput):
        """
        Applies the backward pass to the nearest neightbour upsampling  function
        :param gradwrtoutput: Tensor, containing the gradient of the next layer.
        :return: Tensor of the NNUpsample's gradient 
        """
        # There are no parameters to update in the backward pass of the upsampling layer. We only need to downsample the
        # gradient wrt output to match the shape of the input to this layer by summing the gradients of outputs
        # corresponding to the same input entry.
        # This sum can be implemented as a convolution with a specially crafted filter and parameters.

        dl_ds = gradwrtoutput[0]

        kernel_sum = empty(size=(self.n_channels, self.n_channels, self.scale_factor, self.scale_factor)).double().zero_()
        for i in range(self.n_channels):
            kernel_sum[i, i, :, :] = 1
        dl_dx_previous_layer = conv2d(dl_ds, kernel_sum, stride=self.scale_factor)

        return dl_dx_previous_layer
