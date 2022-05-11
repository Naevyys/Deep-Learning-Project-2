from src.module import Module
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold


def dilate(t, d_h, d_w):
    # Dilate the two last dimensions
    size = list(t.size())
    size[-2] = size[-2] * (d_h + 1) - d_h
    size[-1] = size[-1] * (d_w + 1) - d_w
    d = empty(size=size).double().zero_()
    d_h += 1
    d_w += 1
    if len(size) == 2:
        d[::d_h, ::d_w] = t
    elif len(size) == 3:
        d[:, ::d_h, ::d_w] = t
    elif len(size) == 4:
        d[:, :, ::d_h, ::d_w] = t
    return d


class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
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
        self.dl_dw = empty(size=self.w.size())
        self.dl_db = empty(size=self.bias.size())

        # TODO:
        # - Add initialisation parameter to regulate initialisation of weights and bias to avoid vanishing gradient
        #   (waiting for answer to question to TAs)

    def __convolve_forward(self, x):
        """
        Applies kernels to tensor x.
        :param x: Input tensor. Must be of size (batch_size, in_channels, height, width)
        :return: Output of the convolution.
        """

        batch_size, _, height, width = x.shape

        unfolded = unfold(x, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation)
        wxb = self.w.view(self.out_channels, -1) @ unfolded + self.bias.view(1, -1, 1)
        result = wxb.view(batch_size, self.out_channels,
                          (height - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1,
                          (width - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1)

        return result

    def __convolve_backward(self, x, w, b, out_channels):
        """
        Applies kernel w and bias b to input tensor self.x_previous_layer of size (batch_size, channels, height, width).
        :param x: Input tensor. Must be of size (batch_size, channels, height, width)
        # TODO
        :return: Output of the convolution.
        """

        batch_size, _, height, width = x.size()
        kernel_size = w.size()[-2:]

        unfolded = unfold(x, kernel_size=kernel_size)
        wxb = w.contiguous().view(out_channels, -1) @ unfolded + b.view(1, -1, 1)
        result = wxb.view(batch_size, out_channels, height - kernel_size[0] + 1, width - kernel_size[1] + 1)

        return result

    def forward(self, *inputs):
        self.x_previous_layer = inputs[0]  # Store output of previous layer during forward pass, to be used in the backward pass
        return self.__convolve_forward(self.x_previous_layer)

    def backward(self, *gradwrtoutput):
        # Note: the update is performed by the optimizer

        dl_ds = gradwrtoutput[0][0]  # This has the same shape as the output of our layer

        dl_dw = empty(size=self.w.size()).double().zero_()  # Initialize w gradient tensor with the same shape as w
        dl_db = empty(size=self.bias.size()).double().zero_()  # Initialize bias gradient tensor with same shape as bias

        if self.x_previous_layer is not None:  # No gradient if we don't have the output from the previous layer
            # Shape of dl_ds: (B, out_channels, height_in, width_in)
            # Shape of x_previous_layer: (B, in_channels, height_out, width_out)

            # Handle stride
            # If we have stride > 1, there might be some parts of the input that are not convoluted at all.
            # We assume that stride[a] <= kernel_size[a] for all a, otherwise we have to remove certain parts in the
            # middle of the input image, which is too complicated for us to handle.
            # We also need to dilate the kernel to obtain the correct kernel to convolve over x
            x = self.x_previous_layer
            cut_off_height = (x.size()[-2] - self.kernel_size[0]) % self.stride[0]
            cut_off_width = (x.size()[-1] - self.kernel_size[1]) % self.stride[1]
            if cut_off_height > 0:
                x = x[:, :, :-cut_off_height, :]
            if cut_off_width > 0:
                x = x[:, :, :, :-cut_off_width]
            kernel = dilate(dl_ds, self.stride[0] - 1, self.stride[1] - 1)  # e.g. Dilate by 1 if stride is 2

            # Algo for dl_dw
            # For i in channels of dl_ds (i,e, out_channels):
            #   For j in channels of x_previous_layer (i.e. in_channels):
            #      c = convolve dl_ds[:, i, :, :] with x_previous_layer[:, j, :, :]  # Shape of x is (1, 1, height_kernel, width_kernel)
            #      dl_dw[j, i, :, :] = c

            for batch in range(dl_ds.size()[0]):
                for i in range(self.in_channels):
                    for j in range(self.out_channels):
                        x_conv = x[batch:batch+1, i:i+1, :, :]
                        kernel_conv = kernel[batch:batch+1, j:j+1, :, :]
                        backward_conv_bias = empty(size=(1,)).double().zero_()  # Change that to a scalar zero tensor  # dl_dw will be of size (in_channels, kernel_size_1, kernel_size_2)
                        res = self.__convolve_backward(x_conv, kernel_conv, backward_conv_bias, 1)  # accumulate gradient over the entire batch
                        dl_dw[j:j + 1, i:i + 1, :, :] += res

            if self.has_bias:
                # Compute dl_db
                pass

        self.dl_dw = dl_dw
        self.dl_db = dl_db  # TODO

        #dl_dx_previous_layer = self.w.t().mv(dl_ds)  # TODO

        #return dl_dx_previous_layer

    def param(self):
        raise NotImplementedError  # TODO
