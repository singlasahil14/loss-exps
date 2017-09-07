from spatial_transformer import transformer
import numpy as np, pandas as pd, tensorflow as tf
from collections import namedtuple, defaultdict
import argparse, os, collections
from model import Network
from data_loader import load_data

class _ClassifierModel(Network):
  def __init__(self):
    Network.__init__(self)

  def _localization_network(self, x):
    with tf.variable_scope('localization'):
      x_flat = self._flatten(x)
      x = self._dense(32, name='dense1')(x_flat)
      x = self._tanh(x)

      w2 = tf.get_variable(name='dense2/weights', shape=[32, 6], initializer=tf.zeros_initializer)
      b2 = tf.Variable(initial_value=[1., 0., 0., 0., 1., 0.], name='dense2/biases')
      x = tf.nn.xw_plus_b(x, w2, b2)
      x = self._tanh(x)
    return x

  def _embedding_network(self, x, nonlin='relu'):
    with tf.variable_scope('embeddings'):
      x = self._conv_nonlin(3, 16, nonlin=nonlin, name='conv1')(x, stride=1, padding='SAME')
      x = self._max_pool(x, ksize=2, stride=2, name='pool1')

      x = self._conv_nonlin(3, 16, nonlin=nonlin, name='conv2')(x, stride=1, padding='SAME')
      x = self._max_pool(x, ksize=2, stride=2, name='pool2')

      x = self._flatten(x)
      x = self._dense_nonlin(1024, nonlin=nonlin, name='dense1')(x)
    return x

  def forward(self, x, num_classes, stn=True, nonlin='relu'):
    shape = x.get_shape().as_list()[1:]
    if(stn):
      x_transform = self._localization_network(x)
      out_size = x.get_shape().as_list()[1:]
      x = transformer(x, x_transform, out_size)
      x = tf.reshape(x, [-1]+shape)
    self.embeddings = self._embedding_network(x, nonlin)
    self.logits = self._dense(num_classes, name='logits')(self.embeddings)

  def _conv_nonlin(self, filter_size, out_filters, nonlin='identity', name='conv_nonlin'):
    """Convolution layer with non-linearity."""
    conv_fn = self._conv(filter_size, out_filters)
    def conv_nonlin_fn(inp, stride, padding, batch_norm=True):
      bn_fn = self._batch_norm if batch_norm else tf.nn.identity
      lin_fn = lambda x: bn_fn(conv_fn(x, stride, padding=padding))
      nonlin_fn = self._nonlin_dict[nonlin]

      with tf.variable_scope(name):
        x = nonlin_fn(inp, lin_fn)
      return x
    return conv_nonlin_fn

  def _dense_nonlin(self, out_dim, nonlin='identity', name='dense_nonlin'):
    """FullyConnected layer with non-linearity."""
    dense_fn = self._dense(out_dim)
    def dense_nonlin_fn(inp, batch_norm=True):
      bn_fn = self._batch_norm if batch_norm else tf.nn.identity
      lin_fn = lambda x: bn_fn(dense_fn(x))
      nonlin_fn = self._nonlin_dict[nonlin]
      with tf.variable_scope(name):
        x = nonlin_fn(inp, lin_fn)
      return x
    return dense_nonlin_fn

class LossMinimizer:
  def __init__(self, model_config):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=sess_config)

    (x_train, y_train), (x_test, y_test) = load_data(model_config.dataset)
    self._x_train, self._y_train = x_train, y_train
    self._x_test, self._y_test = x_test, y_test

    num_classes = y_train.shape[1]
    img_shape = list(x_train.shape[1:])
    self._images = tf.placeholder(tf.float32, [None] + img_shape)
    self._labels = tf.placeholder(tf.float32, [None, num_classes])

    model = _ClassifierModel()
    model.forward(self._images, num_classes, stn=model_config.use_stn, nonlin=model_config.nonlin)
    self._logits = model.logits
    self._embeddings = model.embeddings
    self._setup_loss()

  def _setup_loss(self):
    self._cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, labels=self._labels))
    correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._labels, 1))
    self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    optimizer = tf.train.AdamOptimizer()
    self._train_step = optimizer.minimize(self._cross_entropy)

    self._tensor_names = ['cross_entropy', 'accuracy']
    self._tensors_to_fetch = [self._cross_entropy, self._accuracy]

  def _append_metrics(self, metrics_dict, values):
    metrics_dict['iterations'].append(self._iters)
    for name, value in zip(self._tensor_names, values):
      metrics_dict[name].append(value)
    
  def run_optimization(self, num_epochs=500, checkpoint_iters=10, batch_size=128, result_path=None):
    train_metrics = defaultdict(list)
    test_metrics = defaultdict(list)
    os.makedirs(result_path)

    self._sess.run(tf.global_variables_initializer())
    self._iters = 0
    for epoch_i in range(num_epochs):
      i = 0
      while i < len(self._x_train):
        last = min(i+batch_size, len(self._x_train))
        batch_xs = self._x_train[i: last]
        batch_ys = self._y_train[i: last]
        i = last

        feed_dict={self._images: batch_xs, self._labels: batch_ys}
        train_values = self._sess.run([self._train_step] + self._tensors_to_fetch, feed_dict=feed_dict)
        self._append_metrics(train_metrics, train_values[1:])
        if self._iters % checkpoint_iters == 0:
          loss, accuracy = train_values[1], train_values[2]
          print('Iteration: ' + str(self._iters) + ' Loss: ' + str(loss), ' Accuracy: ' + str(accuracy))
        self._iters = self._iters + 1

      feed_dict = {self._images: self._x_test, self._labels: self._y_test}
      test_values = self._sess.run(self._tensors_to_fetch, feed_dict=feed_dict)
      self._append_metrics(test_metrics, test_values)

      if(result_path is not None):
        pd_train_metrics = pd.DataFrame(train_metrics)
        pd_train_metrics.to_csv(os.path.join(result_path, 'train_metrics.csv'))
        pd_test_metrics = pd.DataFrame(test_metrics)
        pd_test_metrics.to_csv(os.path.join(result_path, 'test_metrics.csv'))

      test_cross_entropy, test_accuracy = test_values[0], test_values[1]
      print('End of epoch %d' % epoch_i)
      print('Test Cross Entropy: %.3f, Test Accuracy: %.2f' % (test_cross_entropy, test_accuracy))

def add_arguments(parser):
  parser.add_argument('--dataset', choices=['cluttered-mnist', 'fashion-mnist', 'cifar10', 'cifar100'], default='cluttered-mnist', type=str, 
                      help='Dataset to train the model on (default %(default)s)')
  parser.add_argument('--use-stn', default=False, help='use spatial transformer network', action='store_true')
  parser.add_argument('--nonlin', choices=['relu', 'selu', 'maxout'], default='relu', type=str, help='nonlinearity to use (default %(default)s)')
  parser.add_argument('--num-epochs', default=500, type=int, help='number of epochs to run (default: %(default)s)')
  parser.add_argument('--checkpoint-iters', default=10, type=int, help='number of epochs to run (default: %(default)s)')
  parser.add_argument('--batch-size', default=128, type=int, help='batch size (default %(default)s)')
  parser.add_argument('--result-path', default='result', type=str, help='Directory for storing training and eval logs')
  
def check_arguments(options):
  assert options.num_epochs > 0
  assert options.checkpoint_iters > 0
  assert options.batch_size > 0
  assert not(os.path.exists(options.result_path)), "result dir already exists!"

def main():
  parser = argparse.ArgumentParser()
  add_arguments(parser)

  options = parser.parse_args()
  check_arguments(options)

  model_config_tuple = collections.namedtuple('Model', 'dataset use_stn nonlin')
  model_config = model_config_tuple(dataset=options.dataset, use_stn=options.use_stn, nonlin=options.nonlin)

  loss_minimizer = LossMinimizer(model_config)
  loss_minimizer.run_optimization(num_epochs=options.num_epochs, checkpoint_iters=options.checkpoint_iters, 
                                  batch_size=options.batch_size, result_path=options.result_path)
  
if __name__ == '__main__':
  main()
