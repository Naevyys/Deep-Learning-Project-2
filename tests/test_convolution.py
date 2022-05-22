from unittest import TestCase
import torch
from src.Layers.convolution import Conv2d
from torch.nn.functional import conv2d


def run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias, stride, dilation, padding):
    # Initialize random test input tensor
    if batch_size is None:
        in_size = (in_channels, height, width)
    else:
        in_size = (batch_size, in_channels, height, width)
    x = torch.randn(size=in_size).double()

    # Compute result of our implementation
    tested_conv2d = Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=stride, dilation=dilation, padding=padding)
    actual = tested_conv2d.forward(x)  # Returns a tuple with one element

    # Compute expected result
    weights = tested_conv2d.w  # Retrieve randomly initialized weights from convolution layer
    bias_vals = tested_conv2d.bias if bias else None  # Retrieve randomly initialized bias from convolution layer if any
    expected = conv2d(x, weights, bias=bias_vals, stride=stride, dilation=dilation, padding=padding)

    return actual, expected


def run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels, bias, stride, dilation, padding):
    kernel_size_tup = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride_tup = (stride, stride) if isinstance(stride, int) else stride
    padding_tup = (padding, padding) if isinstance(padding, int) else padding
    dilation_tup = (dilation, dilation) if isinstance(dilation, int) else dilation

    # Initialize random inputs and targets
    if batch_size is None:
        in_size = (in_channels, height, width)
        out_size = (out_channels, (height + 2 * padding_tup[0] - dilation_tup[0] * (kernel_size_tup[0] - 1) - 1) // stride_tup[0] + 1,
                   (width + 2 * padding_tup[1] - dilation_tup[1] * (kernel_size_tup[1] - 1) - 1) // stride_tup[1] + 1)
    else:
        in_size = (batch_size, in_channels, height, width)
        out_size = (batch_size, out_channels, (height + 2 * padding_tup[0] - dilation_tup[0] * (kernel_size_tup[0] - 1) - 1) // stride_tup[0] + 1,
                   (width + 2 * padding_tup[1] - dilation_tup[1] * (kernel_size_tup[1] - 1) - 1) // stride_tup[1] + 1)
    x = torch.randn(size=in_size).double()  # Input
    y = torch.randn(size=out_size).double()  # Target

    # Initialize our convolution and call forward
    tested_conv2d = Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=stride, dilation=dilation, padding=padding)
    tested_conv2d.forward(x)

    # Initialize a torch convolution and call forward
    w, b = tested_conv2d.w, tested_conv2d.bias
    torch_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, stride=stride,
                                 dilation=dilation, padding=padding)
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
    grad_of_loss = torch.autograd.grad(criterion(output, y), output)[0]  # Input to backward pass of our convolution
    torch.set_grad_enabled(False)  # Disabling autograd once we do not need it anymore

    # Compute dl_dw and dl_db using Conv2d.backward() (call Conv2d.forward() first to set self.x_previous_layer)
    tested_conv2d.backward(grad_of_loss)
    dl_dw_actual = tested_conv2d.dl_dw
    dl_db_actual = tested_conv2d.dl_db

    return (dl_dw_actual, dl_dw_expected), (dl_db_actual, dl_db_expected)


def run_backward_test_dl_dx_previous_layer_indirectly(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding, kernel_size2, out_channels2,
                                                      bias2, stride2, dilation2, padding2):
    # In this test, we want to test whether the value dl_dx_previous_layer computed is correct. Since we do not know
    # where to extract this value from torch directly, we instead test our value using two convolutions, and we pass the
    # value computed by the backward pass of the second convolution as argument to the backward pass of the first
    # convolution and verify whether the dl_dw and dl_db values computed for the first convolution are correct.

    kernel_size_tup = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride_tup = (stride, stride) if isinstance(stride, int) else stride
    padding_tup = (padding, padding) if isinstance(padding, int) else padding
    dilation_tup = (dilation, dilation) if isinstance(dilation, int) else dilation

    kernel_size2_tup = (kernel_size2, kernel_size2) if isinstance(kernel_size2, int) else kernel_size2
    stride2_tup = (stride2, stride2) if isinstance(stride2, int) else stride2
    padding2_tup = (padding2, padding2) if isinstance(padding2, int) else padding2
    dilation2_tup = (dilation2, dilation2) if isinstance(dilation2, int) else dilation2

    # Initialize random inputs and targets
    if batch_size is None:
        size_in = (in_channels, height, width)
        size_out1 = (out_channels,
                     (height + 2 * padding_tup[0] - dilation_tup[0] * (kernel_size_tup[0] - 1) - 1) // stride_tup[
                         0] + 1,
                     (width + 2 * padding_tup[1] - dilation_tup[1] * (kernel_size_tup[1] - 1) - 1) // stride_tup[1] + 1)
        size_out2 = (out_channels2,
                     (size_out1[-2] + 2 * padding2_tup[0] - dilation2_tup[0] * (kernel_size2_tup[0] - 1) - 1) //
                     stride2_tup[0] + 1,
                     (size_out1[-1] + 2 * padding2_tup[1] - dilation2_tup[1] * (kernel_size2_tup[1] - 1) - 1) //
                     stride2_tup[1] + 1)
    else:
        size_in = (batch_size, in_channels, height, width)
        size_out1 = (batch_size, out_channels, (height + 2 * padding_tup[0] - dilation_tup[0] * (kernel_size_tup[0] - 1) - 1) // stride_tup[0] + 1,
                              (width + 2 * padding_tup[1] - dilation_tup[1] * (kernel_size_tup[1] - 1) - 1) // stride_tup[1] + 1)
        size_out2 = (batch_size, out_channels2, (size_out1[-2] + 2 * padding2_tup[0] - dilation2_tup[0] * (kernel_size2_tup[0] - 1) - 1) // stride2_tup[0] + 1,
                              (size_out1[-1] + 2 * padding2_tup[1] - dilation2_tup[1] * (kernel_size2_tup[1] - 1) - 1) // stride2_tup[1] + 1)
    x = torch.randn(size=size_in).double()  # Input
    y = torch.randn(size=size_out2).double()  # Target

    # Initialize our convolutions and call forward
    tested_conv2d = Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=stride, dilation=dilation, padding=padding)
    tested_conv2d2 = Conv2d(out_channels, out_channels2, kernel_size2, bias=bias2, stride=stride2, dilation=dilation2, padding=padding2)
    tested_conv2d2.forward(tested_conv2d.forward(x))

    # Initialize a torch convolution and call forward
    w, b = tested_conv2d.w, tested_conv2d.bias
    w2, b2 = tested_conv2d2.w, tested_conv2d2.bias
    torch_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, stride=stride,
                                 dilation=dilation, padding=padding)
    torch_conv.weight, torch_conv.bias = torch.nn.Parameter(w), torch.nn.Parameter(b)
    torch_conv2 = torch.nn.Conv2d(out_channels, out_channels2, kernel_size=kernel_size2, bias=bias2, stride=stride2,
                                 dilation=dilation2, padding=padding2)
    torch_conv2.weight, torch_conv2.bias = torch.nn.Parameter(w2), torch.nn.Parameter(b2)
    torch.set_grad_enabled(True)  # Temporarily enable autograd for testing purposes
    output = torch_conv2(torch_conv(x))

    # Calculate loss using torch MSE loss
    torch_conv.zero_grad()
    torch_conv2.zero_grad()
    criterion = torch.nn.MSELoss()
    loss = criterion(output, y)

    # Compute expected dl_dw and dl_db from torch layer
    loss.backward()
    dl_dw_expected = torch_conv.weight.grad
    dl_db_expected = torch_conv.bias.grad
    dl_dw_expected2 = torch_conv2.weight.grad
    dl_db_expected2 = torch_conv2.bias.grad
    grad_of_loss = torch.autograd.grad(criterion(output, y), output)[0]  # Input to backward pass of our convolution
    torch.set_grad_enabled(False)  # Disabling autograd once we do not need it anymore

    # Compute dl_dw and dl_db using Conv2d.backward() (call Conv2d.forward() first to set self.x_previous_layer)
    tested_conv2d.backward(tested_conv2d2.backward(grad_of_loss))
    dl_dw_actual = tested_conv2d.dl_dw
    dl_db_actual = tested_conv2d.dl_db
    dl_dw_actual2 = tested_conv2d2.dl_dw
    dl_db_actual2 = tested_conv2d2.dl_db

    return (dl_dw_actual, dl_dw_expected), (dl_db_actual, dl_db_expected), (dl_dw_actual2, dl_dw_expected2), (dl_db_actual2, dl_db_expected2)


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
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

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
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

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
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

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
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

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
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

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
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

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
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_int_stride(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 10

        # Convolution parameters for testing
        kernel_size = 3
        out_channels = 3
        bias = False
        stride = 3
        dilation = 1
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

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
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_tuple_stride_unequal_sizes(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 10

        # Convolution parameters for testing
        kernel_size = 3
        out_channels = 3
        bias = False
        stride = (3, 2)
        dilation = 1
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

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
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

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
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

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
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_int_padding(self):
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
        padding = 1

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_tuple_padding_unequal_sizes(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 3
        out_channels = 3
        bias = False
        stride = 1
        dilation = 1
        padding = (1, 2)

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_tuple_padding_unequal_sizes_2(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 4
        out_channels = 3
        bias = False
        stride = 1
        dilation = 1
        padding = (2, 1)

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_int_kernel_with_bias_with_batch_size_1(self):
        # Parameters for test input tensor
        batch_size = 1
        in_channels = 3
        height = 4
        width = 5

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = True
        stride = 1
        dilation = 1
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

        # Compare expected and obtained results
        self.assertTrue(torch.allclose(expected, actual))

    def test_forward_with_int_kernel_with_bias_no_batch_size(self):
        # Parameters for test input tensor
        batch_size = None
        in_channels = 3
        height = 4
        width = 5

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = True
        stride = 1
        dilation = 1
        padding = 0

        # Run the test
        actual, expected = run_forward_test(batch_size, in_channels, height, width, kernel_size, out_channels, bias,
                                            stride, dilation, padding)

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
        padding = 0

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

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
        padding = 0

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

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
        padding = 0

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

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
        padding = 0

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

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
        padding = 0

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

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
        padding = 0

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

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
        padding = 0

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

    def test_backward_dl_dw_no_bias_with_int_padding(self):
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
        padding = 1

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

    def test_backward_dl_dw_no_bias_with_tuple_padding(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 3
        out_channels = 3
        bias = False
        stride = 1
        dilation = 1
        padding = (1, 2)

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

    def test_backward_dl_dw_no_bias_with_tuple_padding_2(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 3
        out_channels = 3
        bias = False
        stride = 1
        dilation = 1
        padding = (2, 1)

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

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
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

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
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

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
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

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
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

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
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

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
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

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
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

    def test_backward_dl_dw_no_bias_with_int_dilation(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 3
        out_channels = 3
        bias = False
        stride = 1
        dilation = 2
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

    def test_backward_dl_dw_no_bias_with_asymmetric_dilation(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 3
        out_channels = 3
        bias = False
        stride = 1
        dilation = (2, 3)
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

    def test_backward_dl_dw_with_bias_with_int_dilation(self):
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
        dilation = 2
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

    def test_backward_dl_dw_with_bias_with_asymmetric_dilation(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 11
        width = 6

        # Convolution parameters for testing
        kernel_size = 3
        out_channels = 3
        bias = True
        stride = 1
        dilation = (3, 2)
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

    def test_backward_dl_dw_with_bias_with_int_padding(self):
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
        padding = 1

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

    def test_backward_dl_dw_with_bias_with_tuple_padding(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 3
        out_channels = 3
        bias = True
        stride = 1
        dilation = 1
        padding = (1, 2)

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

    def test_backward_dl_dw_with_bias_with_tuple_padding_2(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 3
        out_channels = 3
        bias = True
        stride = 1
        dilation = 1
        padding = (2, 1)

        dl_dws, _ = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size, out_channels,
                                                      bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))

    def test_backward_dl_dx_previous_layer_int_kernel_no_stride(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 15
        width = 10

        # Convolution 1 parameters for testing
        kernel_size = 3
        out_channels = 4
        bias = False  # Bias doesn't play a role for dl_dx_previous_layer_computation
        stride = 1
        dilation = 1
        padding = 0

        # Convolution 2 parameters for testing
        kernel_size2 = 2
        out_channels2 = 5
        bias2 = False
        stride2 = 1
        dilation2 = 1
        padding2 = 0

        dl_dws, _, dl_dw2s, _ = run_backward_test_dl_dx_previous_layer_indirectly(batch_size, in_channels, height,
                                                                                  width, kernel_size, out_channels,
                                                                                  bias, stride, dilation, padding, kernel_size2,
                                                                                  out_channels2, bias2, stride2,
                                                                                  dilation2, padding2)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dw2s))

    def test_backward_dl_dw_with_bias_batch_size_1(self):
        # Parameters for test input tensor
        batch_size = 1
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = True
        stride = 1
        dilation = 1
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

    def test_backward_dl_dw_with_bias_no_batch_size_dim(self):
        # Parameters for test input tensor
        batch_size = None
        in_channels = 3
        height = 6
        width = 11

        # Convolution parameters for testing
        kernel_size = 2
        out_channels = 3
        bias = True
        stride = 1
        dilation = 1
        padding = 0

        dl_dws, dl_dbs = run_backward_test_dl_dw_and_dl_db(batch_size, in_channels, height, width, kernel_size,
                                                           out_channels, bias, stride, dilation, padding)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dbs))

    def test_backward_dl_dx_previous_layer_asymmetric_kernel_no_stride(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 15
        width = 10

        # Convolution 1 parameters for testing
        kernel_size = (4, 3)
        out_channels = 4
        bias = False
        stride = 1
        dilation = 1
        padding = 0

        # Convolution 2 parameters for testing
        kernel_size2 = (2, 3)
        out_channels2 = 5
        bias2 = False
        stride2 = 1
        dilation2 = 1
        padding2 = 0

        dl_dws, _, dl_dw2s, _ = run_backward_test_dl_dx_previous_layer_indirectly(batch_size, in_channels, height,
                                                                                  width, kernel_size, out_channels,
                                                                                  bias, stride, dilation, padding, kernel_size2,
                                                                                  out_channels2, bias2, stride2,
                                                                                  dilation2, padding2)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dw2s))

    def test_backward_dl_dx_previous_layer_int_kernel_int_stride(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 15
        width = 10

        # Convolution 1 parameters for testing
        kernel_size = 3
        out_channels = 4
        bias = False
        stride = 2
        dilation = 1
        padding = 0

        # Convolution 2 parameters for testing
        kernel_size2 = 2
        out_channels2 = 5
        bias2 = False
        stride2 = 2
        dilation2 = 1
        padding2 = 0

        dl_dws, _, dl_dw2s, _ = run_backward_test_dl_dx_previous_layer_indirectly(batch_size, in_channels, height,
                                                                                  width, kernel_size, out_channels,
                                                                                  bias, stride, dilation, padding, kernel_size2,
                                                                                  out_channels2, bias2, stride2,
                                                                                  dilation2, padding2)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dw2s))

    def test_backward_dl_dx_previous_layer_int_kernel_asymmetric_stride(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 20
        width = 25

        # Convolution 1 parameters for testing
        kernel_size = 3
        out_channels = 4
        bias = False
        stride = (2, 3)
        dilation = 1
        padding = 0

        # Convolution 2 parameters for testing
        kernel_size2 = 3
        out_channels2 = 5
        bias2 = False
        stride2 = (3, 2)
        dilation2 = 1
        padding2 = 0

        dl_dws, _, dl_dw2s, _ = run_backward_test_dl_dx_previous_layer_indirectly(batch_size, in_channels, height,
                                                                                  width, kernel_size, out_channels,
                                                                                  bias, stride, dilation, padding, kernel_size2,
                                                                                  out_channels2, bias2, stride2,
                                                                                  dilation2, padding2)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dw2s))

    def test_backward_dl_dx_previous_layer_int_kernel_int_dilation(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 15
        width = 10

        # Convolution 1 parameters for testing
        kernel_size = 3
        out_channels = 4
        bias = False  # Bias doesn't play a role for dl_dx_previous_layer_computation
        stride = 1
        dilation = 2
        padding = 0

        # Convolution 2 parameters for testing
        kernel_size2 = 3
        out_channels2 = 4
        bias2 = False
        stride2 = 1
        dilation2 = 2
        padding2 = 0

        dl_dws, _, dl_dw2s, _ = run_backward_test_dl_dx_previous_layer_indirectly(batch_size, in_channels, height,
                                                                                  width, kernel_size, out_channels,
                                                                                  bias, stride, dilation, padding, kernel_size2,
                                                                                  out_channels2, bias2, stride2,
                                                                                  dilation2, padding2)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dw2s))

    def test_backward_dl_dx_previous_layer_int_kernel_int_padding(self):
        # Parameters for test input tensor
        batch_size = 10
        in_channels = 3
        height = 15
        width = 10

        # Convolution 1 parameters for testing
        kernel_size = 3
        out_channels = 4
        bias = False  # Bias doesn't play a role for dl_dx_previous_layer_computation
        stride = 1
        dilation = 1
        padding = (1, 2)

        # Convolution 2 parameters for testing
        kernel_size2 = 3
        out_channels2 = 4
        bias2 = False
        stride2 = 1
        dilation2 = 1
        padding2 = (2, 1)

        dl_dws, _, dl_dw2s, _ = run_backward_test_dl_dx_previous_layer_indirectly(batch_size, in_channels, height,
                                                                                  width, kernel_size, out_channels,
                                                                                  bias, stride, dilation, padding, kernel_size2,
                                                                                  out_channels2, bias2, stride2,
                                                                                  dilation2, padding2)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dw2s))

    def test_backward_dl_dx_previous_layer_int_kernel_with_bias_batch_size_1(self):
        # Parameters for test input tensor
        batch_size = 1
        in_channels = 3
        height = 15
        width = 10

        # Convolution 1 parameters for testing
        kernel_size = 3
        out_channels = 4
        bias = True
        stride = 1
        dilation = 1
        padding = 0

        # Convolution 2 parameters for testing
        kernel_size2 = 2
        out_channels2 = 5
        bias2 = True
        stride2 = 1
        dilation2 = 1
        padding2 = 0

        dl_dws, dl_dbs, dl_dw2s, dl_db2s = run_backward_test_dl_dx_previous_layer_indirectly(batch_size, in_channels, height,
                                                                                  width, kernel_size, out_channels,
                                                                                  bias, stride, dilation, padding, kernel_size2,
                                                                                  out_channels2, bias2, stride2,
                                                                                  dilation2, padding2)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dw2s))
        self.assertTrue(torch.allclose(*dl_dbs))
        self.assertTrue(torch.allclose(*dl_db2s))

    def test_backward_dl_dx_previous_layer_int_kernel_with_bias_no_batch_size_dim(self):
        # Parameters for test input tensor
        batch_size = None
        in_channels = 3
        height = 15
        width = 10

        # Convolution 1 parameters for testing
        kernel_size = 3
        out_channels = 4
        bias = True
        stride = 1
        dilation = 1
        padding = 0

        # Convolution 2 parameters for testing
        kernel_size2 = 2
        out_channels2 = 5
        bias2 = True
        stride2 = 1
        dilation2 = 1
        padding2 = 0

        dl_dws, dl_dbs, dl_dw2s, dl_db2s = run_backward_test_dl_dx_previous_layer_indirectly(batch_size, in_channels, height,
                                                                                  width, kernel_size, out_channels,
                                                                                  bias, stride, dilation, padding, kernel_size2,
                                                                                  out_channels2, bias2, stride2,
                                                                                  dilation2, padding2)

        self.assertTrue(torch.allclose(*dl_dws))
        self.assertTrue(torch.allclose(*dl_dw2s))
        self.assertTrue(torch.allclose(*dl_dbs))
        self.assertTrue(torch.allclose(*dl_db2s))

    # def test_backward_zero_loss_gives_zero_gradient(self):
    #     self.fail()  # TODO

    # TODO forward pass tests:
    #  - Padding (see if we wanna implement that)
    # TODO backward pass tests:
    #  - No bias, with padding (only if padding implemented)
    #  - Verify zero loss gives zero gradient
