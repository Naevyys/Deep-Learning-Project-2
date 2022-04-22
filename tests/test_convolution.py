from unittest import TestCase
import torch
from src.Layers.convolution import Conv2d
from torch.nn.functional import conv2d


def run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias, stride, dilation):
    # Initialize random test input tensor
    x = torch.randn(size=(batch_size, in_channels, height, width)).double()

    # Compute result of our implementation
    tested_conv2d = Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=stride, dilation=dilation)
    actual = tested_conv2d.forward(x)

    # Compute expected result
    weights = tested_conv2d.w  # Retrieve randomly initialized weights from convolution layer
    bias_vals = tested_conv2d.bias if bias else None  # Retrieve randomly initialized bias from convolution layer if any
    expected = conv2d(x, weights, bias=bias_vals, stride=stride, dilation=dilation)

    return actual, expected


class TestConv2d(TestCase):

    def test_forward_with_int_kernel(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 4
        width = 5

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = False
        stride = 1
        dilation = 1

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_tuple_kernel_equal_sizes(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 4
        width = 5

        # Convolution parameters for testing
        kernel_size = (2, 2)
        out_channels = 3
        bias = False
        stride = 1
        dilation = 1

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_tuple_kernel_unequal_sizes(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 4
        width = 5

        # Convolution parameters for testing
        kernel_size = (2, 3)
        out_channels = 3
        bias = False
        stride = 1
        dilation = 1

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_bias(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 4
        width = 5

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = True
        stride = 1
        dilation = 1

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_bias_with_tuple_kernel(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 4
        width = 5

        # Convolution parameters for testing
        kernel_size = (3, 2)
        out_channels = 3
        bias = True
        stride = 1
        dilation = 1

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_int_stride(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 10

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = False
        stride = 3
        dilation = 1

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_tuple_stride_equal_sizes(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 10

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = False
        stride = (2, 2)
        dilation = 1

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_tuple_stride_unequal_sizes(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 10

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = False
        stride = (3, 2)
        dilation = 1

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_int_dilation(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = False
        stride = 1
        dilation = 2

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_tuple_dilation_equal_sizes(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = False
        stride = 1
        dilation = (2, 2)

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_tuple_dilation_unequal_sizes(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = False
        stride = 1
        dilation = (2, 3)

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_backward_dl_dw_and_dl_db_are_correct(self):
        # TODO but not for here, just putting this here as a note:
        #  - Save output from previous layer in forward passes of loss and activation functions as well

        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11
        x = torch.randn(size=(batch_size, in_channels, height, width)).double()  # Input to the layer

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = True
        stride = 1
        dilation = 1

        # Compute dl_dw and dl_db using Conv2d.backward() (call Conv2d.forward() first to set self.x_previous_layer)
        tested_conv2d = Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=stride, dilation=dilation)
        tested_conv2d.forward(x)
        dl_ds = ...  # Generate a random gradient with respect to the loss from the next layer.  # TODO
        tested_conv2d.backward(dl_ds)
        dl_dw_actual, dl_db_actual = tested_conv2d.dl_dw, tested_conv2d.dl_db

        # compute expected dl_dw and dl_db using Tensor.backward() and calling Tensor.grad
        # Have a look at torch.autograd.grad too
        w, b = tested_conv2d.w, tested_conv2d.bias  # TODO: Keep if needed, otherwise remove
        dl_dw_expected, dl_db_expected = ..., ...  # TODO

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))
        self.assertTrue(torch.allclose(dl_db_expected, dl_db_actual))

    # def test_backward_zero_loss_gives_zero_gradient(self):
    #     self.fail()  # TODO
