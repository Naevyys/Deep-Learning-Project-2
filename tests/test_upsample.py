from unittest import TestCase
import torch
from src.Layers.nearest_neighbor_upsample import NNUpsample2d
from src.Layers.convolution import Conv2d
from src.Loss_functions.mse import MSELoss
from torch.nn.functional import interpolate


def run_forward_test(factor, in_size):
    # Initialize random input
    x = torch.randn(size=in_size).double()
    x_torch_upsample = x.unsqueeze(0) if len(in_size) == 3 else x  # Torch upsample expects a 4D tensor for 2D upsample

    # Compute expected output from torch
    expected = interpolate(x_torch_upsample, scale_factor=factor)

    # Compute actual output
    actual = NNUpsample2d(factor).forward(x)

    return actual, expected


def run_backward_test(factor, batch_size, in_channels, height, width, kernel_size, out_channels, bias):
    if batch_size is None:
        in_size = (in_channels, height, width)
        out_size = (out_channels, (height - kernel_size + 1)*factor, (width - kernel_size + 1)*factor)
    else:
        in_size = (batch_size, in_channels, height, width)
        out_size = (batch_size, out_channels, (height - kernel_size + 1)*factor, (width - kernel_size + 1)*factor)
    x = torch.randn(size=in_size).double()  # Initialize input
    y = torch.randn(size=out_size).double()  # Initialize random output

    # Initialise one conv and one upsample with my implementation
    conv = Conv2d(in_channels, out_channels, kernel_size, bias=bias)
    up = NNUpsample2d(factor)

    output = up.forward(conv.forward(x))  # Get output
    mse = MSELoss()
    mse.forward(output, y)  # Compute loss with own MSE
    conv.backward(up.backward(mse.backward()))  # Call backward

    torch.set_grad_enabled(True)  # Temporarily enable to be able to compare with torch

    # Insert batch dim if batch_size is None, because upsample of torch expects a batch size for 2d upsample
    if batch_size is None:
        x = x.unsqueeze(0)
    # Initialize one conv and one upsample from torch with same parameters and weights
    conv_torch = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
    conv_torch.weight, conv_torch.bias = torch.nn.Parameter(conv.w), torch.nn.Parameter(conv.bias)
    up_torch = torch.nn.Upsample(scale_factor=factor)

    output = up_torch(conv_torch(x))  # Get output
    mse_torch = torch.nn.MSELoss()
    loss = mse_torch(output, y)  # Compute loss with torch MSE
    loss.backward()

    torch.set_grad_enabled(False)  # Disable again

    # Get dl_dw and dl_db of own conv and torch conv, return that
    dl_dw_actual, dl_db_actual = conv.dl_dw, conv.dl_db
    dl_dw_expected, dl_db_expected = conv_torch.weight.grad, conv_torch.bias.grad

    return (dl_dw_actual, dl_dw_expected), (dl_db_actual, dl_db_expected)


class TestUpsample2d(TestCase):

    ######################################  FORWARD  ######################################

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

    ######################################  BACKWARD  ######################################

    def test_backward_factor_1(self):
        factor = 1
        batch_size = 10
        height, width = 9, 7
        kernel_size = 3
        in_channels, out_channels = 3, 4
        bias = True

        dl_dws, dl_dbs = run_backward_test(factor, batch_size, in_channels, height, width, kernel_size, out_channels,
                                           bias)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

    def test_backward_factor_2(self):
        factor = 2
        batch_size = 10
        height, width = 9, 7
        kernel_size = 3
        in_channels, out_channels = 3, 4
        bias = True

        dl_dws, dl_dbs = run_backward_test(factor, batch_size, in_channels, height, width, kernel_size, out_channels,
                                           bias)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

    def test_backward_factor_3(self):
        factor = 3
        batch_size = 10
        height, width = 9, 7
        kernel_size = 3
        in_channels, out_channels = 3, 4
        bias = True

        dl_dws, dl_dbs = run_backward_test(factor, batch_size, in_channels, height, width, kernel_size, out_channels,
                                           bias)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

    def test_backward_factor_2_batch_size_1(self):
        factor = 2
        batch_size = 1
        height, width = 9, 7
        kernel_size = 3
        in_channels, out_channels = 3, 4
        bias = True

        dl_dws, dl_dbs = run_backward_test(factor, batch_size, in_channels, height, width, kernel_size, out_channels,
                                           bias)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

    def test_backward_factor_2_no_batch_size_dim(self):
        factor = 2
        batch_size = None
        height, width = 9, 7
        kernel_size = 3
        in_channels, out_channels = 3, 4
        bias = True

        dl_dws, dl_dbs = run_backward_test(factor, batch_size, in_channels, height, width, kernel_size, out_channels,
                                           bias)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))
