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


from files import (TRAIN_X_DIR, TRAIN_Y_DIR, TRAIN_X_PATTERN, TRAIN_Y_PATTERN, TEST_X_DIR,
                   TEST_X_PATTERN,
                   get_path_list, get_path_list_2dir, results_save)


H, W = 28, 28
H2, W2 = H // 2, W // 2

def get_patches(x_img, y_img):
    """Our initial hack works on 28x28 pixel patches from the sample images
    """
    assert x_img.shape == y_img.shape
    h, w = x_img.shape[:2]
    x_patches = np.empty(((h - H) * (w - W), H * W), np.uint8)
    y_patches = np.empty((h - H) * (w - W), np.uint8)
    for iy in xrange(h - H):
        ir = iy * (w - W)
        for ix in xrange(w - W):
            x_patches[ir + ix, :] = x_img[iy:iy + H, ix:ix + W].ravel()
            y_patches[ir + ix] = y_img[iy + H2, ix + W2]
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
    H, W = 420, 540

    path_list = get_path_list_2dir(TRAIN_X_PATTERN, TRAIN_Y_DIR)
    x_list = []
    y_list = []
    size = 0
    for i, (path_x, path_y) in enumerate(path_list):
        print('%3d: %s %s' % (i, path_x, path_y), end=' ')
        x_img = cv2.imread(path_x, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        y_img = cv2.imread(path_y, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        x_patches, y_patches= get_patches(x_img, y_img)
        x_list.append(x_img)
        y_list.append(y_img)
        sz = x_patches.size + y_patches.size
        size += sz
        print('%s %s %.0f %.2f' % (list(x_patches.shape), list(y_patches.shape),
              sz / MBYTE, size / GBYTE))

    x0 = x_list[0]
    y0 = y_list[0]
    for x, y in zip(x_list, y_list)[1:]:
        assert x.shape == x0.shape, (x0.shape, x.shape)
        assert y.shape == y0.shape

    n_total = len(path_list)
    n_train = n_total // 2
    n_test = n_total // 4
    n_validate = n_total - n_train - n_test

    def matrixify(img_list):
        n = len(img_list)
        sz = img_list[0].size
        mtx = np.empty((n, sz), dtype=np.uint8)
        for i, img in enumerate(img_list):
            mtx[i, :] = img.ravel()
        return mtx

    def as_matrix(start, end):
        assert isinstance(start, int), start
        assert isinstance(end, int), end
        return matrixify(x_list[start:end]), matrixify(y_list[start:end])

    train_set = as_matrix(0, n_train)
    test_set = as_matrix(n_train, n_train + n_test)
    valid_set = as_matrix(n_train + n_test, n_total)

    # Load the dataset
    # with gzip.open(dataset, 'rb') as f:
    #     train_set, valid_set, test_set = cPickle.load(f)

    if False:
        for X, y in train_set, valid_set, test_set:
            print('X=%s.%s,y=%s.%s' % (list(X.shape), X.dtype, list(y.shape), y.dtype))
        exit()

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an np.ndarray of 2 dimensions (a matrix) with row's correspond to an example.
    # target is a np.ndarray of 1 dimensions (vector)) that have the same length as the number of
    # rows in the input. It should give the target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

