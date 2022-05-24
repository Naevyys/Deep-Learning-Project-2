from torch import empty
from torch.nn.functional import fold, unfold


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """
    Applies weights and biases to tensor x.
    :param x: Input tensor. Must be of size (batch_size, in_channels, height, width) or (in_channels, height, width)
    :param weight: Weights of size (kernel_size[0], kernel_size[1])
    :param bias: Bias values of size (out_channels,)
    :param stride: Stride value, int or tuple
    :param padding: Padding value, int or tuple
    :param dilation: Dilation value, positive int or tuple
    :return: Output of the convolution.
    """

    # Extract useful data for the computation of shapes etc.
    out_channels, _, kernel_size_0, kernel_size_1 = weight.shape
    height, width = x.shape[-2:]

    is_3D = True if len(x.shape) == 3 else False  # Handle case if input has no batch size
    if is_3D:
        x = x.unsqueeze(0)

    # Handle case of bias is None and int stride & dilation
    bias = bias if bias is not None else empty(size=(out_channels,)).double().zero_()
    stride = (stride, stride) if isinstance(stride, int) else stride
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    padding = (padding, padding) if isinstance(padding, int) else padding

    # Convolve
    unfolded = unfold(x, kernel_size=(kernel_size_0, kernel_size_1), stride=stride, dilation=dilation, padding=padding)
    wxb = weight.contiguous().view(out_channels, -1) @ unfolded + bias.view(1, -1, 1)
    result = wxb.view(-1, out_channels, (height + 2 * padding[0] - dilation[0] * (kernel_size_0 - 1) - 1) // stride[0] + 1,
                      (width + 2 * padding[1] - dilation[1] * (kernel_size_1 - 1) - 1) // stride[1] + 1)

    if is_3D:
        result = result.squeeze(dim=0)

    return result


def dilate(t, d_h, d_w):
    """
    Dilate the last two dimensions of tensor t.
    :param t: Tensor to dilate
    :param d_h: Dilation in height
    :param d_w: Dilation in width
    :return: Dilated tensor
    """

    size = list(t.size())
    size[-2] = size[-2] * (d_h + 1) - d_h
    size[-1] = size[-1] * (d_w + 1) - d_w
    d = empty(size=size).double().zero_()
    d_h += 1
    d_w += 1
    if len(size) == 2:  # TODO: improve code structure here, just take last two dimensions and assign them
        d[::d_h, ::d_w] = t
    elif len(size) == 3:
        d[:, ::d_h, ::d_w] = t
    elif len(size) == 4:
        d[:, :, ::d_h, ::d_w] = t

    return d


def waiting_bar(i, length, loss):
        """
            Simple function that prints a progress/waiting bar + the loss
            :param i: Integer, the current element we are working on
            :param length: Integer, the total number of elements we need to work on
            :param loss: Tuple(Float, Float), The training and validation loss of the system
            :return: Nothing, just print
        """
        left = int(30 * i / length)
        right = 30 - left
        tags = "=" * left
        spaces = " " * right
        print(f"\r[{tags}>{spaces}] Train loss: {loss[0]:.5f} Val loss: {loss[1]:.5f}", sep="", end="", flush=True)