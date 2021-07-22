import numpy as np
import h5py

def flatten(data):
    return data.reshape(data.shape[0], -1).T

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    # classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

train_x_orig, train_y, test_x_orig, test_y = load_data()

train_x_orig_flatten = flatten(train_x_orig)
train_y_flatten = flatten(train_y)
test_x_orig_flatten = flatten(test_x_orig)
test_y_flatten = flatten(test_y)

np.savetxt("datasets/train_activation.csv", train_x_orig_flatten, fmt='%i', delimiter=",")
np.savetxt("datasets/train_labels.csv", train_y_flatten, fmt='%i', delimiter=",")
np.savetxt("datasets/test_activation.csv", test_x_orig_flatten, fmt='%i', delimiter=",")
np.savetxt("datasets/test_labels.csv", test_y_flatten, fmt='%i', delimiter=",")
