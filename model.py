import torch.empty

torch.set_grad_enabled(False)

class Model():
    def __init__(self):
        """
        Instantiates the model class.
        :return: None
        """
        raise NotImplementedError
        # takes no other input

    def load_pretrained_model(self):
        """
        Loads best model from file bestmodel.pth
        :return: None
        """
        raise NotImplementedError

    def train(self, train_input, train_target):
        """
        Trains the model.
        :param train_input: Training data.
        :param train_target: Train targets.
        :return: None
        """
        raise NotImplementedError

    def predict(self, test_input):
        """
        Predicts with the model on the provided input.
        :param test_input: Test input.
        :return: The prediction (torch.Tensor).
        """
        raise NotImplementedError
