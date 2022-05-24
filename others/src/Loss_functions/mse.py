from ...src.module import Module
from torch import empty


class MSELoss(Module):

    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, *inputs):  # Compute loss
        """
        Computes the mean squarred error loss.
        :params inputs: A tuple that should contain the prediction of the model and the targets. 
        :return: A double, the MSE
        """
        self.inputs = inputs
        predicted, target = self.inputs
        normalizing_term = 1
        for dim in predicted.shape:
            normalizing_term *= dim
        self.normalizing_term = normalizing_term
        return (predicted - target).pow(2).sum() / normalizing_term

    def backward(self, *gradwrtoutput):  # Compute derivative of the loss
        """
        Computes gradient of the mean squarred error loss.
        :params gradwrtouput: Is not used here as the loss function is the first function producing
        an output on which we can compute a gradient.  
        :return: Tensor containing the gradient for each image of the batch 
        """
        assert self.inputs is not None, "Cannot compute backward without input! First call forward."
        predicted, target = self.inputs
        return 2 * (predicted - target) / self.normalizing_term
