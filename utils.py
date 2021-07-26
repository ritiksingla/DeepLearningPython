import numpy as np
from numpy import ndarray


def assert_same_shape(array: ndarray, array_grad: ndarray):
    assert (
        array.shape == array_grad.shape
    ), '''
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
        '''.format(
        tuple(array_grad.shape), tuple(array.shape)
    )
    return True


def assert_dim(array: ndarray, dim: int):
    assert (
        array.ndim == dim
    ), '''array should have been {0} dimensional 
    but is instead {1} dimensional'''.format(
        dim, array.ndim
    )
    return True


def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def softmax(X):
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]
    return X
