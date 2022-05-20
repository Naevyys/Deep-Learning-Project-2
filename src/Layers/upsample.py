from src.module import Module
from torch import empty, cat, arange


class Upsample2d(Module):

    def __init__(self, factor):
        """
        Initialize an upsample layer
        :param factor: Upsampling factor
        """

        self.factor = factor

    def forward(self, *inputs):
        """
        Upsample the inputs
        :param inputs: Tensor of size (batch_size, channels, height, width)
        :return: Upsampled input
        """

        self.x_previous_layer = inputs[0]

        out_shape = list(self.x_previous_layer.shape)  # (batch_size, in_channels, height, width)
        out_shape[-2], out_shape[-1] = out_shape[-2] * self.factor, out_shape[-1] * self.factor
        x = self.x_previous_layer.unsqueeze(-2).unsqueeze(-1)
        x = x.tile((1, 1, 1, self.factor, 1, self.factor))
        x = x.reshape(out_shape)
        return x

    def backward(self, *gradwrtoutput):

        # downsample (how?) gradwrtoutput

        raise NotImplementedError
