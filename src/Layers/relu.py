from src.module import Module
from torch import empty, cat, arange


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
        # If condition is satisfied, keep original value, else set to 0
        return x.where(x > 0, empty(size=(1,)).float().zero_())  # Needs a tensor with a float for .where() method

    def backward(self, *gradwrtoutput):  # Compute derivative of activation
        """
        Computes derivative of the relu activation function. Note: The derivative of ReLU is undefined at x = 0, so we
        use the subgradient x' = 0 there.
        :param gradwrtoutput: Tensor w.r.t. which we compute the derivative.
        :return: Derivative of the activation.
        """
        x = gradwrtoutput[0]
        x = x.where(x > 0, empty(size=(1,)).float().zero_())
        x = x.where(x <= 0, empty(size=(1,)).float().fill_(1))
        return x
