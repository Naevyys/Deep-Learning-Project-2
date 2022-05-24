import torch
from torch import empty

torch.set_grad_enabled(False)

class Module(object):
    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *inputs):
        """
        Computes the forward pass in the submodule/subclass of the given layer. 
        :param inputs: Tensor or tuple of tensor which are the inputs to forward to the next layer
        :return: Tensor or tuple of tensors.
        """
        raise NotImplementedError("The forward pass has not been implemented for this layer!") 

    def backward(self, *gradwrtoutput):
        """
        Computes the backward pass in the submodule/subclass of the given layer. 
        :param gradwrtoutput: Tensor or tuple of tensors which come from the previous layer 
        :return: Tensor or tuple of tensors containing the gradient of the loss w.r.t the module's input, backwarded to the previous layer 
        """
        
        raise NotImplementedError("The backward pass has not been implemented for this layer!") 

    def param(self):
        """
        Returns list of pairs of parameter tensor and gradient tensor of the same size. Empty for parameterless modules.
        :return: List of pairs of parameter and gradient tensors.
        """
        return []

    def update_param(self, updated_params):
        """
        The function will update the parameters of the sub modules having parameters. For instance convolution.
        It does nothing if the module does not have parameters. 
        """
        pass 

    def zero_grad(self):
        """
        Zero the gradients of the parameters of the module. Does nothing if the module does not have parameters.
        :return: None
        """
        pass
