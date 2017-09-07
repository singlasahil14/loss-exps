from spatial_transformer import transformer
import numpy as np, pandas as pd, tensorflow as tf
from collections import namedtuple, defaultdict
import argparse, os, collections, json
from model import Network
from data_loader import load_data

tf.set_random_seed(99)
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

  def forward(self, x, stn=True, nonlin='relu'):
    shape = x.get_shape().as_list()[1:]
    if(stn):
      x_transform = self._localization_network(x)
      out_size = x.get_shape().as_list()[1:]
      x = transformer(x, x_transform, out_size)
      x = tf.reshape(x, [-1]+shape)
    self.embeddings = self._embedding_network(x, nonlin)

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
  def __init__(self, model_config, result_path=None):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=sess_config)

    self._result_path = result_path
    if self._result_path is not None:
      os.makedirs(self._result_path)
      config_str = json.dumps(model_config._asdict())
      config_file = os.path.join(self._result_path, 'config')
      config_file_object = open(config_file, 'w')
      config_file_object.write(config_str)

    (x_train, y_train), (x_test, y_test) = load_data(model_config.dataset)
    self._x_train, self._y_train = x_train, y_train
    self._x_test, self._y_test = x_test, y_test

    self._num_classes = y_train.shape[1]
    img_shape = list(x_train.shape[1:])
    self._images = tf.placeholder(tf.float32, [None] + img_shape)
    self._labels = tf.placeholder(tf.float32, [None, self._num_classes])

    model = _ClassifierModel()
    model.forward(self._images, stn=model_config.use_stn, nonlin=model_config.nonlin)
    self._embeddings = model.embeddings
    self._embeddings = tf.nn.l2_normalize(self._embeddings, dim=1)
    self._embedding_size = self._embeddings.get_shape().as_list()[-1]
    with tf.variable_scope('logits'):
      w = tf.get_variable('weights', [self._embedding_size, self._num_classes],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    w = tf.nn.l2_normalize(w, dim=0)
    self._logits = tf.matmul(self._embeddings, w)

    self._LAMBDA = model_config.LAMBDA
    self._ALPHA = model_config.ALPHA
    self._beta_placeholder = tf.placeholder(tf.float32, shape=())
    self._BETA = model_config.BETA
    self._BETA_DECAY = model_config.BETA_DECAY
    self._M = model_config.M
    self._extra_loss = model_config.extra_loss
    self._setup_loss()

  def _amplify_cos(self, cos):
    thetas = tf.acos(cos)
    pi = tf.acos(-1.)
    ks = tf.floor(self._M*(thetas/pi))
    ks = tf.minimum(ks, self._M-1)
    if(self._M==1):
      amplified_cos = cos
    elif(self._M==2):
      amplified_cos = 2.*tf.square(cos) - 1.
    elif(self._M==3):
      amplified_cos = 4.*tf.square(cos) - 3.
      amplified_cos = cos * amplified_cos
    elif(self._M==4):
      amplified_cos = 2*tf.square(cos) - 1.
      amplified_cos = 2.*tf.square(amplified_cos) - 1.
    amplified_cos = (tf.to_float(tf.pow(-1, tf.to_int32(ks)))*amplified_cos) - 2.*tf.to_float(ks)
    return amplified_cos

  def _margin_cos_thetas(self, cos_thetas, labels):
    amplified_cos_thetas = self._amplify_cos(cos_thetas)
    margin_cos_thetas = cos_thetas*(1-labels) + amplified_cos_thetas*labels
    return margin_cos_thetas

  def _center_loss_fn(self, embeddings, labels):
    centers = tf.get_variable(name='centers', shape=[self._num_classes, self._embedding_size],
                              initializer=tf.random_normal_initializer(stddev=0.1), trainable=False)
    label_indices = tf.argmax(self._labels, 1)
    centers_batch = tf.nn.embedding_lookup(centers, label_indices)
    center_loss = self._LAMBDA * tf.nn.l2_loss(embeddings - centers_batch)/tf.to_float(tf.shape(embeddings)[0])
    new_centers = centers_batch - embeddings
    labels_unique, row_indices, counts = tf.unique_with_counts(label_indices)

    centers_update = tf.unsorted_segment_sum(new_centers, row_indices, tf.shape(labels_unique)[0])/tf.to_float(counts)
    centers = tf.scatter_sub(centers, labels_unique, self._ALPHA*centers_update)
    return center_loss

  def _setup_loss(self):
    embeddings_norm = tf.norm(self._embeddings, axis=1, keep_dims=True)
    cos_thetas = self._logits/embeddings_norm
    modified_cos_thetas = self._margin_cos_thetas(cos_thetas, self._labels)
    modified_logits = embeddings_norm*modified_cos_thetas
    margin_logits = (modified_logits + (self._beta_placeholder*self._logits))/(1+self._beta_placeholder)

    self._center_loss = self._center_loss_fn(self._embeddings, self._labels)
    self._cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._logits, labels=self._labels))
    self._margin_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=margin_logits, labels=self._labels))

    if self._extra_loss == 'center':
      self._total_loss = self._cross_entropy + self._center_loss
    elif self._extra_loss == 'sphere':
      self._total_loss = self._margin_cross_entropy
    else:
      self._total_loss = self._cross_entropy

    trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    optimizer = tf.train.AdamOptimizer()
    self._train_step = optimizer.minimize(self._total_loss)

    correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._labels, 1))
    self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    self._tensor_names = ['cross_entropy', 'accuracy', 'margin_cross_entropy', 'center_loss']
    self._tensors_to_fetch = [self._cross_entropy, self._accuracy, self._margin_cross_entropy, self._center_loss]

  def _append_metrics(self, metrics_dict, values):
    metrics_dict['iterations'].append(self._iters)
    for name, value in zip(self._tensor_names, values):
      metrics_dict[name].append(value)
    
  def run_optimization(self, num_epochs=100, checkpoint_iters=10, batch_size=128, result_path=None):
    train_metrics = defaultdict(list)
    test_metrics = defaultdict(list)

    self._sess.run(tf.global_variables_initializer())
    self._iters = 0
    beta = self._BETA
    format_string = 'Iteration: %d, Cross Entropy: %f, Accuracy: %.2f, Margin Cross Entropy: %f, Center Loss: %.3f'
    for epoch_i in range(num_epochs):
      i = 0
      while i < len(self._x_train):
        last = min(i+batch_size, len(self._x_train))
        batch_xs = self._x_train[i: last]
        batch_ys = self._y_train[i: last]
        i = last

        feed_dict={self._images: batch_xs, self._labels: batch_ys, self._beta_placeholder: beta}
        train_values = self._sess.run([self._train_step] + self._tensors_to_fetch, feed_dict=feed_dict)
        self._append_metrics(train_metrics, train_values[1:])
        if self._iters % checkpoint_iters == 0:
          train_cross_entropy, train_accuracy, train_margin_cross_entropy, train_center_loss = train_values[1], train_values[2], train_values[3], train_values[4]
          print(format_string % (self._iters, train_cross_entropy, train_accuracy, train_margin_cross_entropy, train_center_loss))
        self._iters = self._iters + 1
      beta = beta*self._BETA_DECAY

      feed_dict = {self._images: self._x_test, self._labels: self._y_test, self._beta_placeholder: 0.}
      test_values = self._sess.run(self._tensors_to_fetch, feed_dict=feed_dict)
      self._append_metrics(test_metrics, test_values)

      if(self._result_path is not None):
        pd_train_metrics = pd.DataFrame(train_metrics)
        pd_train_metrics.to_csv(os.path.join(self._result_path, 'train_metrics.csv'))
        pd_test_metrics = pd.DataFrame(test_metrics)
        pd_test_metrics.to_csv(os.path.join(self._result_path, 'test_metrics.csv'))

      test_cross_entropy, test_accuracy = test_values[0], test_values[1]
      print('End of epoch %d' % epoch_i)
      print('Test Cross Entropy: %.3f, Test Accuracy: %.2f' % (test_cross_entropy, test_accuracy))

def add_arguments(parser):
  parser.add_argument('--dataset', choices=['cluttered-mnist', 'fashion-mnist', 'cifar10', 'cifar100'], default='cluttered-mnist', type=str, 
                      help='Dataset to train the model on (default %(default)s)')
  parser.add_argument('--use-stn', default=False, help='use spatial transformer network', action='store_true')
  parser.add_argument('--nonlin', choices=['relu', 'selu', 'maxout'], default='relu', type=str, help='nonlinearity to use (default %(default)s)')
  parser.add_argument('--extra-loss', choices=['center', 'a-softmax'], default=None, type=str, help='extra loss to add to the total loss (default None)')
  parser.add_argument('--LAMBDA', default=0.003, type=float, help='constant to multiply with center loss (default %(default)s)')
  parser.add_argument('--ALPHA', default=0.5, type=float, help='constant to update embedding centroids (default %(default)s)')
  parser.add_argument('--BETA', default=1000, type=float, help='constant for a-softmax loss gradient update (default %(default)s)')
  parser.add_argument('--BETA-DECAY', default=0.1, type=float, help='decay constant for a-softmax loss gradient update (default %(default)s)')
  parser.add_argument('--M', choices=[1, 2, 3, 4], default=2, type=int, help='angle multiplier for a-softmax loss (default %(default)s)')
  parser.add_argument('--num-epochs', default=100, type=int, help='number of epochs to run (default %(default)s)')
  parser.add_argument('--checkpoint-iters', default=10, type=int, help='number of epochs to run (default %(default)s)')
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

  model_config_tuple = collections.namedtuple('Model', 'dataset use_stn nonlin extra_loss LAMBDA ALPHA M BETA BETA_DECAY')
  model_config = model_config_tuple(dataset=options.dataset, use_stn=options.use_stn, nonlin=options.nonlin, extra_loss=options.extra_loss, 
                   LAMBDA=options.LAMBDA, ALPHA=options.ALPHA, M=options.M, BETA=options.BETA, BETA_DECAY=options.BETA_DECAY)

  loss_minimizer = LossMinimizer(model_config, result_path=options.result_path)
  loss_minimizer.run_optimization(num_epochs=options.num_epochs, checkpoint_iters=options.checkpoint_iters, 
                                  batch_size=options.batch_size, result_path=options.result_path)
  
if __name__ == '__main__':
  main()
