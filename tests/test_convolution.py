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


def run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels, bias, stride, dilation):
    kernel_size_tup = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride_tup = (stride, stride) if isinstance(stride, int) else stride
    dilation_tup = (dilation, dilation) if isinstance(dilation, int) else dilation

    # Initialize random inputs and targets
    x = torch.randn(size=(batch_size, in_channels, height, width)).double()  # Input
    y = torch.randn(size=(batch_size, out_channels, (height - dilation_tup[0] * (kernel_size_tup[0] - 1) - 1) // stride_tup[0] + 1,
                          (width - dilation_tup[1] * (kernel_size_tup[1] - 1) - 1) // stride_tup[1] + 1)).double()  # Target

    # Initialize our convolution and call forward
    tested_conv2d = Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=stride, dilation=dilation)
    tested_conv2d.forward(x)

    # Initialize a torch convolution and call forward
    w, b = tested_conv2d.w, tested_conv2d.bias
    torch_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, stride=stride,
                                 dilation=dilation)
    torch_conv.weight, torch_conv.bias = torch.nn.Parameter(w), torch.nn.Parameter(b)
    torch.set_grad_enabled(True)  # Temporarily enable autograd for testing purposes
    output = torch_conv(x)

    # Calculate loss using torch MSE loss
    torch_conv.zero_grad()
    criterion = torch.nn.MSELoss()
    loss = criterion(output, y)

    # Compute expected dl_dw and dl_db from torch layer
    loss.backward()
    dl_dw_expected = torch_conv.weight.grad
    dl_db_expected = torch_conv.bias.grad
    grad_of_loss = torch.autograd.grad(criterion(output, y), output)  # Input to backward pass of our convolution
    torch.set_grad_enabled(False)  # Disabling autograd once we do not need it anymore

    # Compute dl_dw and dl_db using Conv2d.backward() (call Conv2d.forward() first to set self.x_previous_layer)
    tested_conv2d.backward(grad_of_loss)
    dl_dw_actual = tested_conv2d.dl_dw
    dl_db_actual = tested_conv2d.dl_db

    return (dl_dw_actual, dl_db_actual), (dl_dw_expected, dl_db_expected)


class TestConv2d(TestCase):

    ######################################  FORWARD  ######################################

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

    def test_forward_with_int_kernel_out_channels_greater_than_in_channels(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 4
        width = 5

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 4
        bias = False
        stride = 1
        dilation = 1

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_int_kernel_out_channels_smaller_than_in_channels(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 4
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

    ######################################  BACKWARD  ######################################

    def test_backward_dl_dw_no_bias_no_params(self):
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
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    def test_backward_dl_dw_no_bias_no_params_out_channels_greater_than_in_channels(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 4
        bias = False
        stride = 1
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    def test_backward_dl_dw_no_bias_no_params_out_channels_smaller_than_in_channels(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 4
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = False
        stride = 1
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    def test_backward_dl_dw_no_bias_no_params_asymmetric_kernel(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = (2, 3)
        out_channels = 3
        bias = False
        stride = 1
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    def test_backward_dl_dw_no_bias_with_int_stride(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = False
        stride = 2
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    def test_backward_dl_dw_no_bias_with_asymmetric_stride(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 3
        out_channels = 3
        bias = False
        stride = (2, 3)
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    def test_backward_dl_dw_no_bias_with_border_case_kernel_stride_pairs(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 15
        width = 10

        # Convolution parameters for testing
        kernel_size = 5
        out_channels = 3
        bias = False
        stride = (3, 2)
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    def test_backward_dl_dw_with_bias_no_params(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = True
        stride = 1
        dilation = 1

        (dl_dw_actual, dl_db_actual), (dl_dw_expected, dl_db_expected) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))
        self.assertTrue(torch.allclose(dl_db_expected, dl_db_actual))

    def test_backward_dl_dw_with_bias_no_params_out_channels_greater_than_in_channels(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 4
        bias = True
        stride = 1
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    def test_backward_dl_dw_with_bias_no_params_out_channels_smaller_than_in_channels(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 4
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = True
        stride = 1
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    def test_backward_dl_dw_with_bias_no_params_asymmetric_kernel(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = (2, 3)
        out_channels = 3
        bias = True
        stride = 1
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    def test_backward_dl_dw_with_bias_with_int_stride(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = True
        stride = 2
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    def test_backward_dl_dw_with_bias_with_asymmetric_stride(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 3
        out_channels = 3
        bias = True
        stride = (2, 3)
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    def test_backward_dl_dw_with_bias_with_border_case_kernel_stride_pairs(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 15
        width = 10

        # Convolution parameters for testing
        kernel_size = 5
        out_channels = 3
        bias = True
        stride = (3, 2)
        dilation = 1

        (dl_dw_actual, _), (dl_dw_expected, _) = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height,
                                                                                   width, kernel_size, out_channels,
                                                                                   bias, stride, dilation)

        self.assertTrue(torch.allclose(dl_dw_expected, dl_dw_actual))

    # def test_backward_zero_loss_gives_zero_gradient(self):
    #     self.fail()  # TODO

    # TODO forward pass tests:
    #  - Padding (see if we wanna implement that)
    # TODO backward pass tests:
    #  - No bias, with padding (only if padding implemented)
    #  - With bias all of the above again
    #  - Verify dl_dx output from the backward is correct
    #  - Verify zero loss gives zero gradient
