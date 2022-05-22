import torch
from torch import empty

torch.set_grad_enabled(False)

class Module(object):
    def __call__(self, *args):
        self.forward(*args)

    def forward(self, *inputs):
        """
        Implementation of the forward pass.
        :param inputs:
        :return: Tensor or tuple of tensors.
        """
        # TODO: complete docstring
        # Note: Implementation of the method is done in each submodule.
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        """
        Implementation of the backward pass.
        :param gradwrtoutput:
        :return: Tensor or tuple of tensors containing the gradient of the loss w.r.t the module's input
        """
        # TODO: complete docstring
        raise NotImplementedError

    def param(self):
        """
        Returns list of pairs of parameter tensor and gradient tensor of the same size. Empty for parameterless modules.
        :return: List of pairs of parameter and gradient tensors.
        """
        return []

    def update_param(self, updated_params):
        """
        The function will update the parameters of the sub modules having parameters. For instance convolution. 
        """
        pass 

    def zero_grad(self):
        """
        Zero the gradients of the parameters of the module.
        :return: None
        """
        pass
