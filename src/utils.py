from torch import empty

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


def pad(t, p_u, p_d, p_l, p_r):
    """
    Pad the last two dimensions of tensor t.
    :param t: Tensor to pad
    :param p_u: Padding up
    :param p_d: Padding down
    :param p_l: Padding left
    :param p_r: Padding right
    :return: Padded tensor
    """

    size = list(t.size())
    size[-2] += p_u + p_d
    size[-1] += p_l + p_r
    p = empty(size=size).double().zero_()

    p_u = p_u if p_u > 0 else None
    p_d = - p_d if p_d > 0 else None
    p_l = p_l if p_l > 0 else None
    p_r = - p_r if p_r > 0 else None

    if len(size) == 2:  # TODO: improve code structure here, just take last two dimensions and assign them
        p[p_u:p_d, p_l:p_r] = t
    elif len(size) == 3:
        p[:, p_u:p_d, p_l:p_r] = t
    elif len(size) == 4:
        p[:, :, p_u:p_d, p_l:p_r] = t

    return p