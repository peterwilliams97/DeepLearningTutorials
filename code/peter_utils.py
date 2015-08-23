"""

"""
from __future__ import division, print_function

import cPickle
import gzip
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T

import cv2

KAPTH1 = '/dev/kaggle0/dirty.docs/code'
KAPTH2 = '/c/peter.dev/kaggle/dirty.docs/code'
for path in KAPTH1, KAPTH2:
    if os.path.exists(path):
        sys.path.insert(0, path)
        print(sys.path)

from utils import lowpriority
from files import (TRAIN_X_DIR, TRAIN_Y_DIR, TRAIN_X_PATTERN, TRAIN_Y_PATTERN, TEST_X_DIR,
                   TEST_X_PATTERN,
                   get_path_list, get_path_list_2dir, results_save)

N_POINTS = 500
H, W = 28, 28
H2, W2 = H // 2, W // 2
D = 2

def get_patches(x_img, y_img):
    """Our initial hack works on 28x28 pixel patches from the sample images
    """
    assert x_img.shape == y_img.shape
    h, w = x_img.shape[:2]
    # x_patches = np.empty(((h - H) * (w - W), H * W), np.uint8)
    # y_patches = np.empty((h - H) * (w - W), np.uint8)
    x_patches, y_patches = [], []
    for iy in xrange(h - H):
        ir = iy * (w - W)
        for ix in xrange(w - W):
            non_white = y_img[iy + H2 - D: iy + H2 + D + 1, ix + W2 - D: ix + W2 + D] < 0x7f
            if not non_white.any():
                continue
            # x_patches[ir + ix, :] = x_img[iy:iy + H, ix:ix + W].ravel()
            # y_patches[ir + ix] = y_img[iy + H2, ix + W2]
            x_patches.append(x_img[iy:iy + H, ix:ix + W].ravel())
            y_patches.append(y_img[iy + H2, ix + W2])
    return x_patches, y_patches


MBYTE = 2 ** 20
GBYTE = 2 ** 30

def load_data(path):
    """Loads the dataset

        path: data directory
    """

    #############
    # LOAD DATA #
    #############

    print('... loading data')

    path_list = get_path_list_2dir(TRAIN_X_PATTERN, TRAIN_Y_DIR)
    x_list = []
    y_list = []
    size = 0
    size2 = 0
    total = 0
    for i, (path_x, path_y) in enumerate(path_list):
        print('%3d: %s %s' % (i, path_x, path_y), end=' ')
        x_img = cv2.imread(path_x, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        y_img = cv2.imread(path_y, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        x_patches, y_patches= get_patches(x_img, y_img)
        x_list.extend(x_patches)
        y_list.extend(y_patches)
        sz = sum(x.size for x in x_patches) + sum(x.size for x in y_patches)
        sz2 = sz / (H * W + 1)
        tot = x_img.size + y_img.size
        size += sz
        size2 += sz2
        total += tot
        orig = x_img.size + y_img.size
        print('[%4d] %.0f MB, %.2f GB (%.2f MB %.4f %.4f)' % (len(x_list), sz / MBYTE, size / GBYTE,
                orig / MBYTE, sz2 / tot, size2 / total))
        if len(x_list) > N_POINTS:
            break

    n_total = len(x_list)
    n_train = n_total // 2
    n_test = n_total // 4
    n_validate = n_total - n_train - n_test

    def matrixify(x_list, y_list):
        n = len(x_list)
        sz = x_list[0].size
        x_mtx = np.empty((n, sz), dtype=np.uint8)
        y_mtx = np.empty(n, dtype=np.uint8)
        assert x_mtx.shape[1] == H * W, (x_mtx.shape[1], H * W)
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            x_mtx[i, :] = x.ravel()
            y_mtx[i] = y
        return x_mtx, y_mtx


    def as_matrix(start, end):
        assert isinstance(start, int), start
        assert isinstance(end, int), end
        return matrixify(x_list[start:end], y_list[start:end])

    train_set = as_matrix(0, n_train)
    test_set = as_matrix(n_train, n_train + n_test)
    valid_set = as_matrix(n_train + n_test, n_total)

    if False:
        # Load the dataset
        dataset='../data/mnist.pkl.gz'
        with gzip.open(dataset, 'rb') as f:
            train_set, valid_set, test_set = cPickle.load(f)

    if False:
        for X, y in train_set, valid_set, test_set:
            print('X=%s.%s,y=%s.%s' % (list(X.shape), X.dtype, list(y.shape), y.dtype))
        exit()

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an np.ndarray of 2 dimensions (a matrix) with row's correspond to an example.
    # target is a np.ndarray of 1 dimensions (vector)) that have the same length as the number of
    # rows in the input. It should give the target to the example with the same index in the input.

    def shared_dataset(data_xy, name_xy, borrow=True):
        """ Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        name_x, name_y = name_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow, name=name_x)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow, name=name_y)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        assert data_x.shape[1] == W * H, (data_x.shape[1], W * H)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set, ('test_set_x', 'test_set_y'))
    valid_set_x, valid_set_y = shared_dataset(valid_set, ('valid_set_x', 'valid_set_y'))
    train_set_x, train_set_y = shared_dataset(train_set, ('train_set_x', 'train_set_y'))

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

print('!' * 300)
lowpriority()
