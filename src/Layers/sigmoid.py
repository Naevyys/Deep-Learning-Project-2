from ...src.module import Module
from torch import empty, cat, arange


class Sigmoid(Module):

    def __init__(self):
        super().__init__()
        self.x_previous_layer = None

    def forward(self, *inputs):  # Compute result of activation
        """
        Applies sigmoid activation function (logistic function) to tensor x
        :param inputs: Tensor, input of the layer.
        :return: Tensor after applying sigmoid to each value of the tensor.
        """
        x = inputs[0].double()
        self.x_previous_layer = x
        return 1 / (1 + (-x).exp())

    def __derivative(self, x):  # Compute derivative of activation
        """
        Computes derivative of the sigmoid activation function.
        :param x: Tensor w.r.t. which we compute the derivative.
        :return: Derivative of the activation.
        """
        return -x.exp() / (-x.exp()+1) ** 2

    def backward(self, *gradwrtoutput):
        dl_dx = gradwrtoutput[0]
        dl_ds = self.__derivative(self.x_previous_layer) * dl_dx if self.x_previous_layer is not None else dl_dx
        return dl_ds

    
    
