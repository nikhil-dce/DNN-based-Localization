""" 
Work in progress

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.client import timeline

from datetime import datetime
import os.path
import re
import time
import ops

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import dataset
import image_preprocessing

import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'data_tb_m_gpu_1',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_integer('print_frequency', 10,
                            """Frequency at which to print""")

tf.app.flags.DEFINE_integer('summary_frequency', 10,
                            """Frequency at which to save summary.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


tf.app.flags.DEFINE_integer('batch_size', 16,
                           """Number of images to process in a batch.""")

tf.app.flags.DEFINE_string('dataset_dir', 'generateTFExamples/examples',
                           """Path to the data directory.""")


# tf.app.flags.DEFINE_string('dataset_subset', 'semantic3D',""" Name of dataset""")
TOWER_NAME = 'tower'




def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def tower_loss(scope,keep_prob):
  """Calculate the total loss on a single tower running the CIFAR model.
  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Get images and labels for CIFAR-10.
  model = model.FRONTEND_VGG_MODIFIED(vgg19_npy_path=FLAGS.vgg_path,trainable=True)

  train_data = dataset.Dataset(FLAGS.dataset_name,'train',FLAGS.dataset_dir)

  images, labels = image_preprocessing.distorted_inputs(train_data)
  print ("LABELS SHAPE -before downsampling is : %s " % labels.get_shape()  )
  # Build inference Graph.
  logits = model.inference(images,keep_prob)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
 
  # return tf.nn.l2_loss(logits)
  _ = model.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % frontend_model.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)
  return total_loss , model



def startSessionAndSaveGraph():
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement = True
    config.log_device_placement=FLAGS.log_device_placement
    sess = tf.Session(config=config)
    sess.run(init)
# 
    # sess = tf.Session(config=tf.ConfigProto(
    #     allow_soft_placement=True,
    #     log_device_placement=FLAGS.log_device_placement))
    # sess.run(init)



    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    raw_input("Press Enter to continue...")


def returnGpu(i):
  if i ==0:
    return FLAGS.gpuOne
  if i ==1:
    return FLAGS.gpuTwo
  if i ==2:
    return FLAGS.gpuThree
  if i ==3:
    return FLAGS.gpuFour

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (frontend_model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * frontend_model.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(frontend_model.INITIAL_LEARNING_RATE_ADAM,
                                    global_step,
                                    decay_steps,
                                    frontend_model.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    # opt = tf.train.GradientDescentOptimizer(lr)

    opt = tf.train.AdamOptimizer(frontend_model.INITIAL_LEARNING_RATE_ADAM)

    print("ADAM Learning rate is %1.10f" % frontend_model.INITIAL_LEARNING_RATE_ADAM)

    with tf.device('/cpu:0'):
      keep_prob = tf.placeholder_with_default(0.5,[], name=None)

    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        gpuId = returnGpu(i)
        with tf.device('/gpu:%d' % gpuId):
          print('---------------------------Tower %d ----------------------------------' % (gpuId))
          with tf.name_scope('%s_%d' % (frontend_model.TOWER_NAME, gpuId)) as scope:
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss, model = tower_loss(scope,keep_prob)
            # startSessionAndSaveGraph()
            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)
            
            # Keep track of the gradients across all towers.
            tower_grads.append(grads)
            # startSessionAndSaveGraph()


    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    print('---------------------------Towers done ----------------------------------')

    grads = average_gradients(tower_grads)


    print("GRADIENTS CALCULATED")

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    # for grad, var in grads:
    #   if grad is not None:
    #     summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # # Apply the gradients to adjust the shared variables.
    # with tf.variable_scope("clipping_gradient"):
    #   grads = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grads]

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # for var in tf.trainable_variables():
    #   if 'filter' in var.op.name:
    #     summaries.append(tf.summary.image(var.op.name, tf.reshape(var, tf.stack([ var.get_shape()[2]*var.get_shape()[3],var.get_shape()[0],var.get_shape()[1] , tf.constant(1)]))))

    print("SUMMARIES SAVED")

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        frontend_model.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    batchnorm_updates_op = tf.group(*batchnorm_updates)

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op,batchnorm_updates_op)
    


    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement = True
    config.log_device_placement=FLAGS.log_device_placement
    sess = tf.Session(config=config)
    sess.run(init)

    print("EVERYTHING IS INITIALIZED")

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    meta_graph_def = tf.train.export_meta_graph(filename=FLAGS.train_dir+'/my-model.meta')

    print("GRAPH IS  SAVED")

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss ])
      # _, loss_value, accuracy_ = sess.run([train_op, loss, model.accuracy])
      # print (labels_val)


      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % FLAGS.print_frequency == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('::step %d, loss = %.10f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % ( step, loss_value,
                             examples_per_sec, sec_per_batch))

        # format_str = ('::step %d, loss = %.10f, accuracy = %.3f (%.1f examples/sec; %.3f '
        #               'sec/batch)')
        # print (format_str % ( step, loss_value, accuracy_,
        #                      examples_per_sec, sec_per_batch))

      if step % FLAGS.summary_frequency == 0:
        summary_str = sess.run(summary_op)
        print("Summary str has been written")
        # print(summary_str)
        summary_writer.add_summary(summary_str, step)
 
      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step,write_meta_graph=False)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  print ("traning directory is %s " %(FLAGS.train_dir))
  if( (FLAGS.patch_size - 2*FLAGS.margin) % 8 != 0):
    print("Please Patch size such that Patch size - 2*margin is divisible by 8... ")
    return

  train()



if __name__ == '__main__':
  tf.app.run()
