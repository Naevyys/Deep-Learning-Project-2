from unittest import TestCase
import torch
from src.Layers.convolution import Conv2d
from torch.nn.functional import conv2d


class TestConv2d(TestCase):

    def test_forward(self):
        # Initialize random test input tensor
        batch_size = 10
        in_channels = 3
        height = 4
        width = 5
        x = torch.randn(size=(batch_size, in_channels, height, width)).double()

        # Set convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = False
        stride = 1

        # Compute result of our implementation
        tested_conv2d = Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=stride)
        actual = tested_conv2d.forward(x)

        # Compute expected result
        weights = tested_conv2d.w  # Retrieve randomly initialized weights from convolution layer
        expected = conv2d(x, weights, bias=None, stride=stride)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))


    # TODO add these tests:
    # - Test tuple kernel size
    #   - height == width
    #   - height =/= width
    # - Test int stride
    # - Test tuple stride
    #   - height == width
    #   - height =/= width
    # - Test bias
    # - Add more tests if more arguments, e.g. padding
