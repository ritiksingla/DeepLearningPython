import numpy as np
from numpy import ndarray
from base import ParamOperation
from utils import assert_dim, assert_same_shape
from conv_utils import (
    _pad_1d,
    _pad_1d_batch,
    _pad_2d,
    _pad_2d_channel,
    _pad_2d_channel_batch,
)


# class Conv1D(object):
#     def __init__(self, param_):
#         '''
#         Parameters
#         ----------
#         param_ : ndarray : 1d numpy array
#             kernel or filter to apply convolution
#         '''
#         assert_dim(param_, 1)
#         self.param_ = param_
#         self.param_len = param_.shape[0]
#         self.param_mid = self.param_len // 2

#     def _conv1d_single_batch(self, input_: ndarray) -> ndarray:

#         # pad the input
#         input_pad = _pad_1d(input_, self.param_mid)

#         # initialize the output
#         output_ = np.zeros_like(input_)

#         # perform the 1d convolution
#         for i in range(input_.shape[0]):
#             output_[i] = np.sum(self.param_ * input_pad[i : i + self.param_len])

#         # ensure shapes didn't change
#         assert_same_shape(input_, output_)
#         return output_

#     def _param_grad_1d_single_batch(
#         self, input_: ndarray, output_grad: ndarray
#     ) -> ndarray:
#         assert_dim(input_, 1)
#         assert_same_shape(input_, output_grad)
#         input_pad = _pad_1d(input_, self.param_mid)
#         param_grad = np.zeros_like(self.param_)

#         # computing gradient w.r.t params
#         for i in range(input_.shape[0]):
#             param_grad += input_pad[i : i + self.param_len] * output_grad[i]

#         # ensure shapes didn't change
#         assert_same_shape(param_grad, self.param_)

#         return param_grad

#     def _input_grad_1d_single_batch(
#         self, input_: ndarray, output_grad: ndarray
#     ) -> ndarray:
#         assert_dim(input_, 1)
#         assert_same_shape(input_, output_grad)
#         input_pad = _pad_1d(input_, self.param_mid)

#         # Zero padded 1 dimensional convolution
#         input_grad = np.zeros_like(input_pad)

#         for i in range(input_.shape[0]):
#             input_grad[i : i + self.param_len] += output_grad[i] * self.param_
#         input_grad = input_grad[self.param_mid : -self.param_mid]
#         assert_same_shape(input_grad, input_)

#         return input_grad

#     def forward(self, input_: ndarray) -> ndarray:

#         '''
#         Apply 1d convolution
#         Parameters
#         ----------

#         input_ : ndarray : 1d numpy array
#                 input array to apply kernel on
#         '''
#         if input_.ndim == 1:
#             self.output_ = self._conv1d_single_batch(input_)
#         elif input_.ndim == 2:
#             self.output_ = np.stack(
#                 [
#                     self._conv1d_single_batch(input_single_batch)
#                     for input_single_batch in input_
#                 ]
#             )
#         else:
#             raise ValueError(
#                 'input must be 1D vector or batches of 1D vectors but got {}D vector'
#                 .format(input_.ndim)
#             )

#         self.input_ = input_
#         return self.output_

#     def backward(self, output_grad: ndarray = None) -> ndarray:
#         if output_grad is None:
#             output_grad = np.ones_like(self.input_)
#         else:
#             assert_same_shape(self.input_, output_grad)
#         if self.input_.ndim == 1:
#             self.param_grad_ = self._param_grad_1d_single_batch(
#                 self.input_, output_grad
#             )
#             self.input_grad_ = self._input_grad_1d_single_batch(
#                 self.input_, output_grad
#             )
#         elif self.input_.ndim == 2:
#             self.param_grad_ = np.sum(
#                 np.stack(
#                     [
#                         self._param_grad_1d_single_batch(input_i, output_grad_i)
#                         for (input_i, output_grad_i) in zip(self.input_, output_grad)
#                     ]
#                 ),
#                 axis=0,
#             )
#             self.input_grad_ = np.stack(
#                 [
#                     self._input_grad_1d_single_batch(input_i, output_grad_i)
#                     for (input_i, output_grad_i) in zip(self.input_, output_grad)
#                 ]
#             )
#         assert_same_shape(self.input_grad_, self.input_)
#         assert_same_shape(self.param_grad_, self.param_)
#         return self.input_grad_


class Conv2D_OP(ParamOperation):
    def __init__(self, param_: ndarray):
        '''
        Parameters
        ----------
        param_ : ndarray : 2d numpy array of size = (C_out, C_in, n_h, n_w),
                where C_in = number of channels in input,
                c_out = Number of channels in output,
                n_h = height of kernel, n_w = width of kernel,
                kernel or filter to apply convolution
        '''
        assert_dim(param_, 4)
        super().__init__(param_)
        self.c_out = param_.shape[0]
        self.c_in = param_.shape[1]
        self.n_h = param_.shape[2]
        self.n_w = param_.shape[3]

    def _get_image_patches(self, input_: ndarray) -> ndarray:
        '''
        Get Image Patches to perform operation with kernel (pooling, convolution, etc.)
        Parameters
        ----------
        input_ : ndarray of size = (batch_size, C_in, H, W)

        Returns
        ----------
        patches : ndarray of size = (batch_size, H * W, C_in, n_h, n_w)
        '''
        assert_dim(input_, 4)
        input_pad = _pad_2d_channel_batch(input_, self.n_h // 2, self.n_w // 2)
        patches = []
        for h in range(input_.shape[2]):
            for w in range(input_.shape[3]):
                patch = input_pad[:, :, h : h + self.n_h, w : w + self.n_w]
                patches.append(patch)
        return np.stack(patches).transpose(1, 0, 2, 3, 4)

    def _param_grad_2d_single_batch(
        self, input_: ndarray, output_grad: ndarray
    ) -> ndarray:
        '''
        Finds gradient of output loss w.r.t parameters of ith layer
        for a single batch (with channels) :
        (dL/d_{input_{i + 1}}) * (d_{input_{i + 1}}/d_{param_{i}})

        Parameters
        ----------
        input_ : ndarray of size = (C_in, H, W)
        output_grad : ndarray of size = (C_out, H, W)

        Returns
        ----------
        param_grad : ndarray of size = (C_out, C_in, n_h, n_w)
        '''
        assert_dim(input_, 3)
        assert_dim(output_grad, 3)

        # pad the input
        input_pad = _pad_2d_channel(input_, self.n_h // 2, self.n_w // 2)
        param_grad = np.zeros_like(self.param_)

        # computing gradient w.r.t params
        for c in range(output_grad.shape[0]):
            for i in range(output_grad.shape[1]):
                for j in range(output_grad.shape[2]):
                    param_grad += (
                        input_pad[:, i : i + self.n_h, j : j + self.n_w]
                        * output_grad[c][i][j]
                    )
        # ensure shapes didn't change
        assert_same_shape(param_grad, self.param_)

        return param_grad

    def _input_grad_2d_single_batch(
        self, input_: ndarray, output_grad: ndarray
    ) -> ndarray:
        '''
        Finds gradient of output loss w.r.t inputs of ith layer
        for a single batch (with channels) :
        (dL/d_{input_{i + 1}}) * (d_{input_{i + 1}}/d_{input_{i}})

        Parameters
        ----------
        input_ : ndarray of size = (C_in, H, W)
        output_grad : ndarray of size = (C_out, H, W)

        Returns
        ----------
        input_grad : ndarray of size = (C_in, H, W)
        '''
        assert_dim(input_, 3)
        assert_dim(output_grad, 3)

        # pad the input
        input_pad = _pad_2d_channel(input_, self.n_h // 2, self.n_w // 2)

        input_grad = np.zeros_like(input_pad)

        for c in range(output_grad.shape[0]):
            for i in range(output_grad.shape[1]):
                for j in range(output_grad.shape[2]):
                    input_grad[:, i : i + self.n_h, j : j + self.n_w] += (
                        output_grad[c][i][j] * self.param_[c]
                    )
        input_grad = input_grad[
            :, (self.n_h // 2) : -(self.n_h // 2), (self.n_w // 2) : -(self.n_w // 2)
        ]
        assert_same_shape(input_grad, input_)

        return input_grad

    def forward(self, input_: ndarray) -> ndarray:
        '''
        Applies 2D Convolution (with padding = 'same') (with channels and batches)

        Parameters
        ----------
        input_ : ndarray of size = (batch_size, C_in, H, W)

        Returns
        ----------
        output_ : ndarray of size = (batch_size, C_out, H, W)
        '''

        if input_.ndim == 3:
            input_ = input_.reshape(1, -1)
        elif input_.ndim != 4:
            raise ValueError(
                'input must be 3D vector or batches of 3D vectors but got {}D vector'
                .format(input_.ndim)
            )

        assert input_.shape[1] == self.c_in, 'Input channels must be same as of param'

        image_patches = self._get_image_patches(input_)

        # initialize the output
        output_ = np.zeros(
            (input_.shape[0], self.c_out, input_.shape[2], input_.shape[3])
        )

        # perform the 2 convolution
        for i_batch in range(input_.shape[0]):
            for c in range(self.c_out):
                output_[i_batch][c] = np.sum(
                    np.reshape(
                        image_patches[i_batch] * self.param_[c],
                        newshape=(input_.shape[2], input_.shape[3], -1),
                    ),
                    axis=2,
                )
        self.input_ = input_
        self.output_ = output_
        return self.output_

    def backward(self, output_grad: ndarray = None) -> ndarray:
        '''
        Backpropagate the gradients

        Parameters
        ----------
        output_grad : ndarray of size = size of output from current layer
            gradients w.r.t. input from future layers : dL/d_input_{i + 1}

        Attributes
        ----------
        input_grad_ : ndarray of size = size of input to current layer
            dL/d_input_{i}, input gradients for back propagation

        param_grad_ : ndarray of size = size of parameters in current layer
            parameter gradient for gradient descent

        Returns
        ----------
        input_grad_
        '''
        if output_grad is None:
            output_grad = np.ones_like(self.output_)
        else:
            assert_same_shape(self.output_, output_grad)

        if self.input_.ndim == 3:
            self.param_grad_ = self._param_grad_2d_single_batch(
                self.input_, output_grad
            )
            self.input_grad_ = self._input_grad_2d_single_batch(
                self.input_, output_grad
            )
        elif self.input_.ndim == 4:
            self.param_grad_ = np.sum(
                np.stack(
                    [
                        self._param_grad_2d_single_batch(input_i, output_grad_i)
                        for (input_i, output_grad_i) in zip(self.input_, output_grad)
                    ]
                ),
                axis=0,
            )
            self.input_grad_ = np.stack(
                [
                    self._input_grad_2d_single_batch(input_i, output_grad_i)
                    for (input_i, output_grad_i) in zip(self.input_, output_grad)
                ]
            )
        assert_same_shape(self.input_grad_, self.input_)
        assert_same_shape(self.param_grad_, self.param_)
        return self.input_grad_
