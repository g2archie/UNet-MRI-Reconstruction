#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

# from six.moves import xrange  # pylint: disable=redefined-builtin

import h5py


def extract_images(filename, imageName, maximg = -1):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    fData = h5py.File(filename, 'r')
    inData = fData.get(imageName)
    if '.h5' in filename:
      print("Data loaded!")
      data = numpy.array(inData)
      print("size of data is {}".format(data.shape))
      return data
    else:

      num_images = inData.shape[0]
      rows = inData.shape[1]
      cols = inData.shape[2]
      time_series = inData.shape[3]
      print("> Loaded file:",filename," of shape: ",num_images, rows, cols, time_series)
      data = numpy.array(inData)
      if maximg != -1:
          data = data[:maximg,:,:,:]
          print("> Loaded file:", filename, " reshaped to: ", data.shape[0], rows, cols, time_series)
  
      # data = data.reshape(num_images, rows, cols, time_series)
      return data


# def dense_to_one_hot(labels_dense, num_classes=10):
#  """Convert class labels from scalars to one-hot vectors."""
#  num_labels = labels_dense.shape[0]
#  index_offset = numpy.arange(num_labels) * num_classes
#  labels_one_hot = numpy.zeros((num_labels, num_classes))
#  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#  return labels_one_hot




class DataSet(object):
    def __init__(self, images, true):
        """Construct a DataSet"""

        assert images.shape[0] == true.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape,
                                                   true.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # assert images.shape[3] == 1
        # images = images.reshape(images.shape[0],
        #                         images.shape[1],images.shape[2])
        # true = true.reshape(true.shape[0],
        #                         true.shape[1],true.shape[2])

        #    Maybe -1 to zero mean
        #    images = numpy.multiply(10.0,images)
        #      # Convert from [0, 255] -> [0.0, 1.0].
        #      images = images.astype(numpy.float32)
        #      images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._true = true
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def true(self):
        return self._true

    @property
    def grad(self):
        return self._grad

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._true = self._true[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._true[start:end]


def read_data_sets( images_recon_path, images_true_path ):
    class DataSets(object):
        pass

    dataSets = DataSets()

    print('Start loading data.')
    trueSet = extract_images(images_true_path, 'imagesTrue')
    imagesSet = extract_images(images_recon_path, 'imagesRecon', trueSet.shape[0])


    dataSets.data = DataSet(imagesSet, trueSet)
    print("Data loaded.")

    return dataSets
