import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS
import h5py
import time
import sys

tf.app.flags.DEFINE_string('train_dir', 'events_summary/run_4',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                           """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('save_pred_every', 2,
                           """Save summary frequency""")
tf.app.flags.DEFINE_integer('print_pred_every', 20,
                           """Print loss every steps""")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 30
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0    # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1# Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-6   # Initial learning rate.
# INITIAL_LEARNING_RATE_ADAM = 1e-4   # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.99, name="batch_norm"):
        with tf.variable_scope(name), tf.device('/cpu:0'):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum, 
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)

class RegressionModel:
  def __init__(self,  trainable=True):
    self.trainable = trainable


  def build(self, images):
    """
    Define model architecture
    """
    tf.summary.image("local",tf.expand_dims(images[:,:,:,0],-1))
    tf.summary.image("prior",tf.expand_dims(images[:,:,:,1],-1))
    self.conv1 =  self.conv_layer(images, 2, 32, "conv1",filter_size=5,padding="SAME")
    self.conv1 =  batch_norm(name='bn_conv1')(self.conv1)
    self.conv1 = self.max_pool(self.conv1,"max_pool_1")

    # print(self.conv1.get_shape())


    self.conv2 =  self.conv_layer(self.conv1, 32, 32, "conv2",filter_size=5,padding="SAME")
    self.conv2 =  batch_norm(name='bn_conv2')(self.conv2)
    self.conv2 = self.max_pool(self.conv2,"max_pool_2")
    # print(self.conv2.get_shape())

    self.fc_1 = self.fc_layer(self.conv2,125*125*32,3,"fc")
    # print(self.fc_1.get_shape())



  def inference(self,images):
    self.build(images)
    print "______________________________"
    print "Network Built"
    return self.fc_1



  def loss(self,output,label):
    with tf.variable_scope("lossss"):
    # tf.add_to_collection('losses', cross_entropy_mean)
      squared_error = tf.losses.mean_squared_error(output , label)

      tf.add_to_collection('losses', squared_error)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')




  def return_train_op(self,total_loss, global_step):
    """Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Calculate the learning rate schedule.
    num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    opt = tf.train.GradientDescentOptimizer(INITIAL_LEARNING_RATE)

    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    grads = opt.compute_gradients(total_loss)
    
    tf.summary.scalar('learning_rate', lr)


    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    batchnorm_updates_op = tf.group(*batchnorm_updates)

    # Group all updates to into a single train op.
    self.train_op = tf.group(apply_gradient_op, variables_averages_op,batchnorm_updates_op)

    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)
      



  def train(self):
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    images = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,500, 500,2))
    label = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,3))
    output = self.inference(images)
    total_loss = self.loss(output,label)
    tf.summary.scalar("loss_value",total_loss)
    self.return_train_op(total_loss,global_step)

    saver = tf.train.Saver(tf.global_variables())
    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)

    print("EVERYTHING IS INITIALIZED")

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    meta_graph_def = tf.train.export_meta_graph(filename=FLAGS.train_dir+'/my-model.meta')
    self.load_dataset(num_items=10000)
    print("GRAPH IS  SAVED")
    self.indices = np.arange(self.num_items)
    np.random.shuffle(self.indices)
    # print self.indices

    for step in range(FLAGS.max_steps):
        start_time = time.time()
        batch_indices = self.minibatch_indices(step)

        # print batch_indices
        minibatch = self.load_minibatch(batch_indices)
        # print minibatch[0].shape
        # print minibatch[1].shape
        feed_dict = { images : minibatch[0], label:minibatch[1] }
        
        if step % FLAGS.save_pred_every == 0:
            loss_value,_, summary = sess.run([total_loss, self.train_op , summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            # save(saver, sess, args.snapshot_dir, step)
        else:
            loss_value, _ = sess.run([total_loss, self.train_op], feed_dict=feed_dict)
        duration = time.time() - start_time
        if step % FLAGS.print_pred_every == 0:
            print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

  def minibatch_indices(self,step):
      counter_start = step * FLAGS.batch_size % self.num_items
      counter_end  = (step+1) * FLAGS.batch_size % self.num_items
      if counter_end > counter_start:
        batch_indices = self.indices[counter_start : counter_end    ]
      else:
        batch_indices = np.zeros(FLAGS.batch_size,dtype = np.int)
        batch_indices[:counter_end] = self.indices[ : counter_end    ]
        batch_indices[counter_end:] = self.indices[:-counter_start    ]

        np.random.shuffle(self.indices)
        # print self.indices

      return batch_indices

  def max_pool(self, bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

  def conv_layer(self, bottom, in_channels, out_channels, name,dilation_factor=1,filter_size=3,padding="SAME"):
    with tf.variable_scope(name) as scope:
        filt, conv_biases = self.get_conv_var(filter_size, in_channels, out_channels, name)
        if dilation_factor == 1:
          conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding=padding)
          bias = tf.nn.bias_add(conv, conv_biases)
        else:
          atrous_conv = tf.nn.atrous_conv2d(bottom, filt, dilation_factor, padding=padding)
          bias = tf.nn.bias_add(atrous_conv, conv_biases)


        relu = tf.nn.relu(bias)
        # relu_bn = tf.contrib.layers.batch_norm(relu, is_training=True, scope=scope,reuse=True)

        return relu


  def fc_layer(self, bottom, in_size, out_size, name):
    with tf.variable_scope(name):

        weights, biases = self.get_fc_var(in_size, out_size, name)

        x = tf.reshape(bottom, [-1, in_size])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc

  def get_conv_var(self, filter_size, in_channels, out_channels, name):
    with tf.device('/cpu:0'):
      initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
      filters = tf.get_variable(name=name + "_filters",initializer=initial_value)

      initial_value = tf.truncated_normal([out_channels], .0, .001)
      biases =  tf.get_variable(name=name + "_biases",initializer=initial_value) 

    return filters, biases

  def get_fc_var(self, in_size, out_size, name):
    with tf.device('/cpu:0'):
      initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
      weights = tf.get_variable(name=name + "_weights",initializer=initial_value)

      initial_value = tf.truncated_normal([out_size], .0, .001)
      biases = tf.get_variable(name=name + "_biases",initializer=initial_value)

    return weights, biases

  def load_dataset(self,hdf5Filename="/media/data_raid/dnn_localization/localization_dataset/annarbor_dataset_dnn.hdf5",num_items=100):
    b = h5py.File(hdf5Filename,"r")
    if not num_items:
      num_items = b['prior_map'].shape[0]
      if num_items%20:

        print "Error"
        sys.exit()
    self.local_map = b['local_map'][:num_items/20]
    self.prior_map = b['prior_map'][:num_items]
    self.output = b['output'][:num_items]
    self.num_items = num_items
    print  "local_map shape " , self.local_map.shape
    print "prior_map shape " , self.prior_map.shape
    print "outputs shape " , self.output.shape
    print "Num of examples(augmented)" , self.num_items

  def load_minibatch(self, indices):
    images = np.zeros( (indices.size , 500,500 , 2 )  )
    output = np.zeros( (indices.size , 3 )  )

    for counter, i in enumerate(indices):
      images[counter,:,:,0] = np.pad(np.reshape(self.local_map[i/20],(200,200) ),[[150,150],[150,150]],'constant', constant_values=(0, 0)) 
      images[counter,:,:,1] = np.reshape(self.prior_map[i],(500,500) )
      output[counter,:] = self.output[i]
    return images,output





def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  print ("traning directory is %s " %(FLAGS.train_dir))
  model = RegressionModel()
  model.train()




if __name__ == '__main__':
  tf.app.run()