from ...src.module import Module
from torch import empty


class ReLU(Module):

    def __init__(self):
        super().__init__()
        self.x_previous_layer = None

    def forward(self, *inputs):  
        """
        Applies relu activation function to tensor inputs
        :param inputs: Tensor, input of the layer.
        :return: Tensor after applying relu to each value of the tensor.
        """
        x = inputs[0].double()
        self.x_previous_layer = x
        # If condition is satisfied, keep original value, else set to 0
        return x.where(x > 0, empty(size=(1,)).double().zero_())  # Needs a tensor with a double for .where() method

    def __derivative(self, x):  
        """
        Computes derivative of the relu activation function. Note: The derivative of ReLU is undefined at x = 0, so we
        use the subgradient x' = 0 there.
        :param x: Tensor w.r.t. which we compute the derivative.
        :return: Derivative of the activation.
        """
        # x.where replaces the entries where the conditions are FALSE 
        x = x.where(x > 0, empty(size=(1,)).double().zero_())
        x = x.where(x <= 0, empty(size=(1,)).double().fill_(1))
        return x

    def backward(self, *gradwrtoutput):
        """
        Applies the backward pass to the relu activation function
        :param gradwrtoutput: Tensor, containing the gradient of the next layer
        :return: Tensor of the relus's gradient
        """
        dl_dx = gradwrtoutput[0]
        dl_ds = self.__derivative(self.x_previous_layer) * dl_dx if self.x_previous_layer is not None else dl_dx
        return dl_ds
