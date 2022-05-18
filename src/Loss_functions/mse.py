from ...src.module import Module


class MSELoss(Module):

    def __init__(self):
        super().__init__()

    def forward(self, *inputs):  # Compute loss
        predicted, target = inputs
        return (predicted - target).pow(2).sum()

    def backward(self, *gradwrtoutput):  # Compute derivative of the loss
        predicted, target = gradwrtoutput
        return 2 * (predicted - target)
