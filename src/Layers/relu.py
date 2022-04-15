from src.module import Module
import torch
from torch import empty


class ReLU(Module):

    def __init__(self):
        super().__init__()

    def __relu(self, x):
        """
        Applies relu activation function to tensor x
        :param x: Tensor, input of the layer.
        :return: Tensor after applying relu to each value of the tensor.
        """
        return torch.where(x < 0, torch.tensor(0, dtype=x.dtype), x)

    def forward(self, *inputs):
        return self.__relu(*inputs)

    def backward(self, *gradwrtoutput):
        raise NotImplementedError  # TODO
