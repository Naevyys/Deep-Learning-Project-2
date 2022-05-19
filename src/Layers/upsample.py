from src.module import Module
from torch import empty, cat, arange


class Upsample(Module):

    def __init__(self, factor):
        """
        Initialize an upsample layer
        :param factor: Upsampling factor
        """

        self.factor = factor

    def forward(self, *inputs):

        # upsample

        raise NotImplementedError

    def backward(self, *gradwrtoutput):

        # downsample (how?) gradwrtoutput

        raise NotImplementedError
