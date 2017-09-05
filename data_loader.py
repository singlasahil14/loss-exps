import mnist_reader
import numpy as np
import tensorflow as tf
import os, gzip

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28, 28, 1)
    return images, labels

def load_data(dataset):
  if(dataset=='fashion-mnist'):
    x_train, y_train = load_mnist('data/fashion', kind='train')
    x_test, y_test = load_mnist('data/fashion', kind='t10k')
  elif(dataset=='cluttered-mnist'):
    mnist_cluttered = np.load('data/mnist_sequence1_sample_5distortions5x5.npz')
    x_train, y_train = mnist_cluttered['X_train'], mnist_cluttered['y_train']
    x_valid, y_valid = mnist_cluttered['X_valid'], mnist_cluttered['y_valid']
    x_test, y_test = mnist_cluttered['X_test'], mnist_cluttered['y_test']
    x_train = np.concatenate((x_train, x_valid))
    y_train = np.concatenate((y_train, y_valid))
    x_train = x_train.reshape(-1, 40, 40, 1)
    x_test = x_test.reshape(-1, 40, 40, 1)
  elif(dataset=='cifar10'):
    (x_train, y_train), (x_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()
  elif(dataset=='cifar100'):
    (x_train, y_train), (x_test, y_test) = tf.contrib.keras.datasets.cifar100.load_data()
  else:
    raise ValueError('dataset not found')
  return (x_train, y_train), (x_test, y_test)