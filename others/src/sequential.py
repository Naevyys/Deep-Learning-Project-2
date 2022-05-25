from torch import empty, cat, arange
from .module import Module

class Sequential(Module):

    def __init__(self, *layers):
        """
        Initiate the class that serves as a container for the different layers of the model.
        : params layers: tuple of Modules (Conv, ReLU, Upsampling etc.)
        :return: None
        """

        self.layers = [x for x in layers]

    def __call__(self, args):
        return self.forward(args)
    
    def forward(self, x):
        """
        Compute the forward pass of the model
        : params x: Tensor, the data which the model needs to predict
        :return predictions: Tensor, the model's predictions
        """
        # Iterate on the layers
        predictions = x
        for layer in self.layers:
            predictions = layer.forward(predictions)

        return predictions

    def backward(self, loss):
        """
        Compute the backward pass of the model.
        : params loss: Tensor, the loss computed from the model's predictions
        :return: None 
        """

        # Iterate on the layers
        d_loss = loss 
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss)

    def param(self): 
        """
        Collect all the parameters of the model in tuples (parameter and derivative of the paramerter)
        :return: A list of tuples of tensors containing the parameters of the model
        """
        return [layer.param() for layer in self.layers]

    def update_param(self, updated_params): 
        """
        Update the parameters of the layers after the SGD
        : params updated_params: A list of new/updated parameters for each layer
        :return: None 
        """

        for layer, param in zip(self.layers, updated_params):
            layer.update_param(param)

    def zero_grad(self):
        """
        Zero the gradients of the module's parameters. 
        :return: None
        """
        for layer in self.layers:
            layer.zero_grad()