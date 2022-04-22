from src.module import Module
from torch import empty, cat, arange


class Sigmoid(Module):

    def __init__(self):
        super().__init__()

    def forward(self, *inputs):  # Compute result of activation
        """
        Applies sigmoid activation function (logistic function) to tensor x
        :param x: Tensor, input of the layer.
        :return: Tensor after applying sigmoid to each value of the tensor.
        """
        x = inputs[0]
        return 1 / (1 + (-x).exp())

    def backward(self, *gradwrtoutput):  # Compute derivative of activation
        """
        Computes derivative of the sigmoid activation function.
        :param gradwrtoutput: Tensor w.r.t. which we compute the derivative.
        :return: Derivative of the activation.
        """
        x = gradwrtoutput[0]
        return x.exp() / (1 + x.exp()) ** 2
