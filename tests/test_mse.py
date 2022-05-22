from unittest import TestCase
from src.Loss_functions.mse import MSELoss
import torch


class TestMSELoss(TestCase):
    def test_forward(self):
        batch_size = 5
        channels = 3
        height = 5
        width = 7

        output = torch.randn(size=(batch_size, channels, height, width)).double()
        target = torch.randn(size=(batch_size, channels, height, width)).double()

        criterion_torch = torch.nn.MSELoss()
        expected_loss = criterion_torch(output, target)

        criterion = MSELoss()
        actual_loss = criterion.forward(output, target)

        self.assertTrue(torch.allclose(expected_loss, actual_loss))

    def test_backward(self):
        batch_size = 5
        channels = 3
        height = 5
        width = 7

        input = torch.randn(size=(batch_size, channels, height, width)).double()
        target = torch.randn(size=(batch_size, channels, height, width)).double()

        conv = torch.nn.Conv2d(channels, channels, kernel_size=1)
        conv.weight = torch.nn.Parameter(torch.ones(size=(channels, channels, 1, 1)).double())

        torch.set_grad_enabled(True)  # Temporarily enable autograd for testing purposes
        output = conv(input)
        conv.zero_grad()
        criterion_torch = torch.nn.MSELoss()
        loss = criterion_torch(output, target)
        loss.backward()
        expected_grad_of_loss = torch.autograd.grad(criterion_torch(output, target), output)[0]  # Input to backward pass of our convolution
        torch.set_grad_enabled(False)  # Disabling autograd once we do not need it anymore

        criterion = MSELoss()
        criterion.forward(output, target)
        actual_loss_grad = criterion.backward()

        self.assertTrue(torch.allclose(expected_grad_of_loss, actual_loss_grad))
