import numpy as np
from numpy import ndarray
from utils import assert_dim, assert_same_shape


def _pad_1d(input_: ndarray, num: int) -> ndarray:
    '''
    Pads single example of 1-dimensional vector with zeros

    Parameters
    ----------
    input_ : ndarray of size = (len,)

    num : int
        padding size

    Returns
    ----------
    z : ndarray of size = (len + 2 * num,)
    '''
    assert_dim(input_, 1)
    if num == 0:
        return input_
    z = np.array([0])
    z = np.repeat(z, num)
    return np.concatenate([z, input_, z])


def _pad_1d_batch(input_: ndarray, num: int) -> ndarray:
    '''
    Pads multiple batches of 1-dimensional vector with zeros

    Parameters
    ----------
    input_ : (batch_size, len)

    num : int
        padding size

    Returns
    ----------
    z : ndarray of size = (batch_size, len + 2 * num)
    '''
    assert_dim(input_, 2)
    if num == 0:
        return input_
    z = np.stack([_pad_1d(input_i, num) for input_i in input_])
    return z


def _pad_2d(input_: ndarray, num_h: int, num_w: int) -> ndarray:
    '''
    Pads single example of 2-dimensional vector with zeros

    Parameters
    ----------
    input_ : (H, W)

    num_h : int
        padding size along dim = 0
    num_w : int
        padding size along dim = 1

    Returns
    ----------
    z : ndarray of size = (H + 2 * num_h, W + 2 * num_w)
    '''
    assert_dim(input_, 2)
    if num_w == 0 and num_h == 0:
        return input_
    input_pad = _pad_1d_batch(input_, num_w)
    other = np.zeros((num_h, input_.shape[1] + 2 * num_w))
    z = np.concatenate([other, input_pad, other])
    return z


def _pad_2d_channel(input_: ndarray, num_h: int, num_w: int) -> ndarray:
    '''
    Pads single example of 2-dimensional vector (with channels) with zeros,

    Parameters
    ----------
    input_ : (channels, H, W)

    num_h : int
        padding size along dim = 0
    num_w : int
        padding size along dim = 1

    Returns
    ----------
    z : ndarray of size = (channels, H + 2 * num_h, W + 2 * num_w)
    '''
    assert_dim(input_, 3)
    if num_w == 0 and num_h == 0:
        return input_
    z = np.stack([_pad_2d(input_i, num_h, num_w) for input_i in input_])
    return z


def _pad_2d_channel_batch(input_: ndarray, num_h: int, num_w: int) -> ndarray:
    '''
    Pads multiple examples of 2-dimensional vector (with channels) with zeros,

    Parameters
    ----------
    input_ : (batch_size, channels, H, W)

    num_h : int
        padding size along dim = 0
    num_w : int
        padding size along dim = 1

    Returns
    ----------
    z : ndarray of size = (batch_size, channels, H + 2 * num_h, W + 2 * num_w)
    '''
    assert_dim(input_, 4)
    if num_w == 0 and num_h == 0:
        return input_
    z = np.stack([_pad_2d_channel(input_i, num_h, num_w) for input_i in input_])
    return z
