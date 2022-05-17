from src.module import Module
from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
from src.utils import conv2d, dilate, pad


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
        self.dl_dw = empty(size=self.w.size()).double().zero_()
        self.dl_db = empty(size=self.bias.size()).double().zero_()

    def forward(self, *inputs):
        self.x_previous_layer = inputs  # Store output of previous layer during forward pass, to be used in the backward pass
        outputs_forward = []
        for i in self.x_previous_layer:
            outputs_forward.append(conv2d(i, self.w, bias=self.bias, stride=self.stride, dilation=self.dilation))
        return tuple(outputs_forward)

    def backward(self, *gradwrtoutput):
        # Note: the update is performed by the optimizer

        assert self.x_previous_layer is not None, "Cannot perform backward pass if no forward pass was performed first!"
        assert len(gradwrtoutput) == len(self.x_previous_layer), "Number of inputs to the backward pass does not match " \
                                                                 "the number of inputs from the forward pass."

        dl_dx_previous_layer = []

        for (x, dl_ds) in zip(self.x_previous_layer, gradwrtoutput):
            dl_dw = empty(size=self.w.size()).double().zero_()  # Initialize w gradient tensor with the same shape as w
            dl_db = empty(size=self.bias.size()).double().zero_()  # Initialize bias gradient tensor with same shape as bias

            # Prepare the backward pass kernel according to stride
            kernel = dilate(dl_ds, self.stride[0] - 1, self.stride[1] - 1)  # e.g. Dilate by 1 if stride is 2


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

            for batch in range(dl_ds.size()[0]):
                for i in range(self.in_channels):
                    for j in range(self.out_channels):
                        x_conv = x[batch:batch+1, i:i+1, :cut_off_height, :cut_off_width]
                        kernel_conv = kernel[batch:batch+1, j:j+1, :, :]
                        res = conv2d(x_conv, kernel_conv)  # accumulate gradient over the entire batch
                        dl_dw[j:j + 1, i:i + 1, :, :] += res

            if self.has_bias:
                dl_db[:] = dl_ds.sum(dim=(0, 2, 3))

            # We accumulate the gradients
            self.dl_dw += dl_dw
            self.dl_db += dl_db

            # Algo for dl_dx_previous_layer
            # - Dilate dl_ds by stride - 1 == variable named kernel
            # - Pad dl_ds by 1 + stride - 1 = stride
            # - Turn w 180 degrees, i.e. flip up-down and left-right
            # - Convolve upside-down w over padded and dilated dl_ds

            dl_ds_processed = pad(kernel, self.kernel_size[0] - 1, self.kernel_size[0] - 1, self.kernel_size[1] - 1, self.kernel_size[1] - 1)
            dl_dx_p_l = empty(size=x.size()).double().zero_()

            for batch in range(dl_ds.size()[0]):
                for i in range(self.out_channels):
                    for j in range(self.in_channels):
                        dl_ds_conv = dl_ds_processed[batch:batch+1, i:i+1, :, :]
                        w_rotated = self.w[i:i+1, j:j+1, :, :]
                        w_rotated[:, :, :, :] = w_rotated[0, 0, :, :].flipud().fliplr()
                        res = conv2d(dl_ds_conv, w_rotated)
                        dl_dx_p_l[batch:batch + 1, j:j + 1, :cut_off_height, :cut_off_width] += res
            dl_dx_previous_layer.append(dl_dx_p_l)
        return tuple(dl_dx_previous_layer)

    def param(self):
        raise NotImplementedError  # TODO
