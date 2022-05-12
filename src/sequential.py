from torch import empty, cat, arange

class Sequential():

    def __init__(self, *layers):
        """
        Initiate the class that serves as a container for the different layers of the model
        : params layers: Modules (Conv, ReLU, Upsampling etc.)
        :return: None
        """

        self.layers = [x for x in layers]

    def __call__(self, *args):
        self.forward(*args)
    
    def forward(self, x):
        """
        Compute the forward pass of the model
        : params x: Tensor, the data which the model needs to predict
        :return predictions: Tensor, the model's predictions
        """

        raise NotImplementedError

        predictions = x 
        return predictions

    def backward(self, loss):
        """
        Compute the backward pass of the model
        : params loss: Tensor, the loss computed from the model's predictions
        :return: List containg each layer's gradient
        """

        raise NotImplementedError