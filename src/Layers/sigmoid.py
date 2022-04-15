from src.module import Module
import torch
from torch import empty


class Sigmoid(Module):

    def __init__(self):
        super().__init__()

    def __sigmoid(self, x):
        """
        Applies sigmoid activation function (logistic function) to tensor x
        :param x: Tensor, input of the layer.
        :return: Tensor after applying sigmoid to each value of the tensor.
        """
        return 1 / (1 + torch.exp(-x))

    def forward(self, *inputs):
        return self.__sigmoid(*inputs)

    def backward(self, *gradwrtoutput):
        raise NotImplementedError  # TODO
