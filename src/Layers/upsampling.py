from src.module import Module
from convolution import Conv2d
from upsample import Upsample


class Upsampling(Module):
    def __init__(self, factor, in_channels, out_channels, **convargs):
        """
        Initialize an upsampling block layer, which is composed of an Upsample layer followed by a Conv2d layer.
        :param factor: Upsampling factor for Upsample layer
        :param in_channels: Number of channels of the samples input to the Upsampling layer (same as for Conv2d layer)
        :param out_channels: Number of channels of the output of the Upsampling layer (same as for Conv2d layer)
        :param convargs: Additional, optional arguments for the convolution layer: stride, dilation, bias.
                         Note that kernel_size and padding are computed automatically from the arguments given to keep
                         the output size of the Conv2d layer the same as the output size of the Upsample layer.
        """

        assert set(convargs.keys()).issubset({"stride", "dilation", "bias"}), "Please pass only arguments in the set " \
                                                                              "{stride, dilation, bias} to the Conv2d!"

        self.upsample = Upsample(factor)

        # Compute the kernel size and padding of Conv2d to have the output size match the output size of Upsample
        kernel_size = ...  # TODO
        padding = ...  # TODO

        self.conv2d = Conv2d(in_channels, out_channels, kernel_size, padding=padding, **convargs)
        self.x_previous_layer = None

    def forward(self, *inputs):
        self.x_previous_layer = inputs[0]
        return self.conv2d(self.upsample(*inputs))

    def backward(self, *gradwrtoutput):
        return self.upsample.backward(self.conv2d.backward(*gradwrtoutput))

    def param(self):
        return [self.upsample.param(), self.conv2d.param()]
        # TODO Quentin: this gives a list with an empty and a non-empty list, check if this is what you want/need

    def update_param(self, updated_params):
        self.conv2d.update_param(updated_params)
