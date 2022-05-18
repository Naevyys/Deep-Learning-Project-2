from ...src.module import Module
from torch import empty, cat, arange
from ...src.utils import conv2d, dilate, pad


class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        """
        Initialize 2d convolution layer. We consider in our implementation that groups=1.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride of the kernel, default is 1.
        :param dilation: Dilation of the kernel, default is 1.
        :param bias: Whether to include a bias term or not. Default True.
        """

        # Verify input types and shapes
        assert isinstance(in_channels, int), "in_channels must be an integer!"
        assert isinstance(out_channels, int), "out_channels must be an integer!"
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), "kernel_size must be an int or a tuple!"
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "kernel_size tuple must be of length 2!"
            assert isinstance(kernel_size[0], int) and isinstance(kernel_size[1], int), "kernel_size tuple value must be ints!"
        assert isinstance(stride, int) or isinstance(stride, tuple), "stride must be an int or a tuple!"
        if isinstance(stride, tuple):
            assert len(stride) == 2, "stride tuple must be of length 2!"
            assert isinstance(stride[0], int) and isinstance(stride[1], int), "stride tuple value must be ints!"
        assert isinstance(padding, int) or isinstance(padding, tuple), "padding must be an int or a tuple!"
        if isinstance(padding, tuple):
            assert len(padding) == 2, "padding tuple must be of length 2!"
            assert isinstance(padding[0], int) and isinstance(padding[1], int), "padding tuple value must be ints!"
        assert isinstance(dilation, int) or isinstance(dilation, tuple), "dilation must be an int or a tuple!"
        if isinstance(dilation, tuple):
            assert len(dilation) == 2, "dilation tuple must be of length 2!"
            assert isinstance(dilation[0], int) and isinstance(dilation[1], int), "dilation tuple value must be ints!"
        assert isinstance(bias, bool), "bias must be a boolean!"

        super().__init__()

        # Set layer parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.has_bias = bias

        # Initialize w and bias
        self.w = empty(size=(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])).double().random_() / 2**53
        if bias:
            self.bias = empty(size=(self.out_channels,)).double().random_() / 2**53
        else:
            self.bias = empty(size=(self.out_channels,)).double().zero_()

        # Initialize parameters that do not have values yet.
        self.x_previous_layer = None
        self.dl_dw = empty(size=self.w.size()).double().zero_()
        self.dl_db = empty(size=self.bias.size()).double().zero_()

    def forward(self, *inputs):
        self.x_previous_layer = inputs[0]  # Store output of previous layer during forward pass, to be used in the backward pass
        return conv2d(self.x_previous_layer, self.w, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def backward(self, *gradwrtoutput):
        # Note: the update is performed by the optimizer

        assert self.x_previous_layer is not None, "Cannot perform backward pass if no forward pass was performed first!"

        dl_ds = gradwrtoutput[0]
        x = self.x_previous_layer

        dl_dw = empty(size=self.w.size()).double().zero_()  # Initialize w gradient tensor with the same shape as w
        dl_db = empty(size=self.bias.size()).double().zero_()  # Initialize bias gradient tensor with same shape as bias

        # Shape of dl_ds: (B, out_channels, height_in, width_in)
        # Shape of x_previous_layer: (B, in_channels, height_out, width_out)

        # Handle stride
        # If we have stride > 1, there might be some parts of the input that are not convoluted at all.
        # We assume that stride[a] <= kernel_size[a] for all a, otherwise we have to remove certain parts in the
        # middle of the input image, which is too complicated for us to handle.

        cut_off_height = (x.size()[-2] - self.kernel_size[0]) % self.stride[0]
        cut_off_width = (x.size()[-1] - self.kernel_size[1]) % self.stride[1]
        cut_off_height = - cut_off_height if cut_off_height > 0 else None
        cut_off_width = - cut_off_width if cut_off_width > 0 else None

        # Algo for dl_dw
        # For i in channels of dl_ds (i,e, out_channels):
        #   For j in channels of x_previous_layer (i.e. in_channels):
        #      c = convolve dl_ds[:, i, :, :] with x_previous_layer[:, j, :, :]  # Shape of x is (1, 1, height_kernel, width_kernel)
        #      dl_dw[j, i, :, :] = c

        x_exp = x.unsqueeze(2)
        kernel_exp = dl_ds.unsqueeze(2)

        for batch in range(kernel_exp.size()[0]):
            kernel_conv = kernel_exp[batch]
            x_conv = x_exp[batch, :, :, :cut_off_height, :cut_off_width]
            res = conv2d(x_conv, kernel_conv, dilation=self.stride, stride=self.dilation, padding=self.padding)  # Dilate to handle stride, stride to handle dilation
            dl_dw += res.transpose(0, 1)

        if self.has_bias:
            dl_db[:] = dl_ds.sum(dim=(0, 2, 3))

        # We accumulate the gradients
        self.dl_dw += dl_dw
        self.dl_db += dl_db

        # Prepare the backward pass kernel according to stride
        kernel = dilate(dl_ds, self.stride[0] - 1, self.stride[1] - 1)  # e.g. Dilate by 1 if stride is 2

        # Algo for dl_dx_previous_layer
        # - Dilate dl_ds by stride - 1 == variable named kernel
        # - Pad dl_ds by 1 + stride - 1 = stride
        # - Turn w 180 degrees, i.e. flip up-down and left-right
        # - Convolve upside-down w over padded and dilated dl_ds

        dl_dx_previous_layer = empty(size=x.size()).double().zero_()

        w_rotated = self.w.transpose(0, 1).flipud().fliplr()
        res = conv2d(kernel, w_rotated, padding=((self.kernel_size[0] - self.padding[0]) * self.dilation[0] - self.dilation[0],
                                                 (self.kernel_size[1] - self.padding[1]) * self.dilation[1] - self.dilation[1]),
                     dilation=self.dilation)
        dl_dx_previous_layer[:, :, :cut_off_height, :cut_off_width] += res
        return dl_dx_previous_layer

    def param(self):
        """
        Returns a list of pairs of parameters. For the the convolution it returns first the weights and their derivatives
        and, then the second element of the list is the bias and its derivative. 
        : returns: The list of parameter described above. 
        """
        return [(self.w, self.dl_dw), (self.bias, self.dl_db)]

    def update_param(self, updated_params):
        """
        Simply updates the parameters of the convolution after the SGD.
        :params updated_params: A list containing first the weights, and second the bias
        :return: None
        """
        self.w = updated_params[0]
        self.bias = updated_params[0]
