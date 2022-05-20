from unittest import TestCase
import torch
from src.Layers.upsample import Upsample2d
from torch.nn.functional import upsample


def run_forward_test(factor, in_size):
    # Initialize random input
    x = torch.randn(size=in_size).double()
    x_torch_upsample = x.unsqueeze(0) if len(in_size) == 3 else x  # Torch upsample expects a 4D tensor for 2D upsample

    # Compute expected output from torch
    expected = upsample(x_torch_upsample, scale_factor=factor)

    # Compute actual output
    actual = Upsample2d(factor).forward(x)

    return actual, expected


class TestUpsample2d(TestCase):
    def test_forward_factor_1(self):
        factor = 1
        batch_size = 10
        in_channels = 3
        height = 7
        width = 5
        in_size = (batch_size, in_channels, height, width)

        out = run_forward_test(factor, in_size)
        self.assertTrue(torch.allclose(*out))

    def test_forward_factor_2(self):
        factor = 2
        batch_size = 10
        in_channels = 3
        height = 7
        width = 5
        in_size = (batch_size, in_channels, height, width)

        out = run_forward_test(factor, in_size)
        self.assertTrue(torch.allclose(*out))

    def test_forward_factor_3(self):
        factor = 3
        batch_size = 10
        in_channels = 3
        height = 7
        width = 5
        in_size = (batch_size, in_channels, height, width)

        out = run_forward_test(factor, in_size)
        self.assertTrue(torch.allclose(*out))

    def test_forward_factor_2_batch_size_1(self):
        factor = 2
        batch_size = 1
        in_channels = 3
        height = 7
        width = 5
        in_size = (batch_size, in_channels, height, width)

        out = run_forward_test(factor, in_size)
        self.assertTrue(torch.allclose(*out))

    def test_forward_factor_2_no_batch_size_dim(self):
        factor = 2
        in_channels = 3
        height = 7
        width = 5
        in_size = (in_channels, height, width)  # No batch size dimension

        out = run_forward_test(factor, in_size)
        self.assertTrue(torch.allclose(*out))
