#!/usr/bin/env python
# coding: utf-8
import numpy as np
import h5py

def write_to_h5(np_array, file_name, dict_name):
    
    f = h5py.File(file_name, 'w')
    f[dict_name] = np_array
    f.close()

test_truth = h5py.File('testData_truth.mat', 'r')
test_input = h5py.File('testData_tga_rot_fullSet.mat', 'r')

train_truth = h5py.File('trainData_truth.mat', 'r')
train_input = h5py.File('trainData_tga_rot.mat', 'r')

test_truth_array = test_truth.get('imagesTrue')[()]
test_input_array = test_input.get('imagesRecon')[()]

train_truth_array = train_truth.get('imagesTrue')[()]
train_input_array = train_input.get('imagesRecon')[()]

test_truth.close()
test_input.close()
train_truth.close()
train_input.close()

full_input_array = np.concatenate((train_input_array, test_input_array), axis=0)
full_truth_array = np.concatenate((train_truth_array, test_truth_array), axis=0)

np.random.seed(102)
indices = np.random.permutation(full_input_array.shape[0])

testing_idx, validation_idx, training_idx = indices[:500], indices[500:1000], indices[1000:]

input_train, input_test, input_validation = full_input_array[training_idx], full_input_array[testing_idx], full_input_array[validation_idx]
truth_train, truth_test, truth_validation = full_truth_array[training_idx], full_truth_array[testing_idx], full_truth_array[validation_idx]

write_to_h5(input_train, 'input_train.h5', 'imagesRecon')
write_to_h5(input_test, 'input_test.h5', 'imagesRecon')
write_to_h5(input_validation, 'input_validation.h5', 'imagesRecon')

write_to_h5(truth_train, 'truth_train.h5', 'imagesTrue')
write_to_h5(truth_test, 'truth_test.h5', 'imagesTrue')
write_to_h5(truth_validation, 'truth_validation.h5', 'imagesTrue')



