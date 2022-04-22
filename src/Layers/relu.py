from src.module import Module
import torch
from torch import empty


class ReLU(Module):

    def __init__(self):
        super().__init__()

    def forward(self, *inputs):  # Compute activation
        """
        Applies relu activation function to tensor inputs
        :param x: Tensor, input of the layer.
        :return: Tensor after applying relu to each value of the tensor.
        """
        x = inputs[0]
        return torch.where(x <= 0, torch.tensor(0, dtype=x.dtype), x)  # TODO: Change function used to match restrictions of the project

    def backward(self, *gradwrtoutput):  # Compute derivative of activation
        """
        Computes derivative of the relu activation function. Note: The derivative of ReLU is undefined at x = 0, so we
        use the subgradient x' = 0 there.
        :param gradwrtoutput: Tensor w.r.t. which we compute the derivative.
        :return: Derivative of the activation.
        """
        x = gradwrtoutput[0]
        return torch.where(x <= 0, torch.tensor(0, dtype=x.dtype), 1)  # TODO: Change function used to match restrictions of the project
