import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS
import h5py
import time
import sys
import re
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

NUMBER_GPU = 2

tf.app.flags.DEFINE_string('train_dir', '/media/data_raid/nikhil/events_summary/run_2',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 200000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', NUMBER_GPU,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                           """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('save_pred_every', 2,
                           """Save summary frequency""")
tf.app.flags.DEFINE_integer('print_pred_every', 20,
                           """Print loss every steps""")

PRIOR_SIZE = 500
LOCAL_SIZE = 200

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
          self.images = images
          tf.summary.image("local",tf.expand_dims(images[:,:,:,0],-1))
          tf.summary.image("prior",tf.expand_dims(images[:,:,:,1],-1))
          self.conv1 =  self.conv_layer(images, 2, 32, "conv1",filter_size=5,padding="SAME")
          self.conv1 =  batch_norm(name='bn_conv1')(self.conv1)
          self.conv1 = self.max_pool(self.conv1,"max_pool_1")

          print(self.conv1.get_shape())


          self.conv2 =  self.conv_layer(self.conv1, 32, 32, "conv2",filter_size=5,padding="SAME")
          self.conv2 =  batch_norm(name='bn_conv2')(self.conv2)
          self.conv2 = self.max_pool(self.conv2,"max_pool_2")
          # print(self.conv2.get_shape())

          self.fc_1 = self.fc_layer(self.conv2,125*125*32,3,"fc")
          # print(self.fc_1.get_shape())
          self.output = self.fc_1

     def inference(self,images):
          self.build(images)
          print "______________________________"
          print "Network Built"
          return self.output



     def loss(self, output, label):

          with tf.variable_scope("output_loss"):
               self.label = label
    
               # squared_error = tf.losses.mean_squared_error(output , label)
               weights = [1 ,1, 1]
               x_error = tf.reduce_mean( tf.square(output[:,0] - label[:,0]) )
               y_error = tf.reduce_mean( tf.square(output[:,1] - label[:,1]) )
               theta_error1 =  tf.minimum( tf.square(output[:,2] - label[:,2]), tf.square(output[:,2] + 3.6 - label[:,2])  ) 
               theta_error1 =  tf.minimum( theta_error1 , tf.square(output[:,2] - 3.6 - label[:,2])  ) 
               theta_error=  tf.reduce_mean( theta_error1 ) 
          
               tf.summary.scalar("x_error", x_error)
               tf.summary.scalar("y_error", y_error)
               tf.summary.scalar("theta_error", theta_error)
      
               weighted_error = weights[0]*x_error + weights[1]*y_error + weights[2]*theta_error
               tf.add_to_collection('losses', weighted_error)

               # beta = 0.001
               # lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars
               # if 'bias' not in v.name ]) * beta
               # Add L2 loss later in the future
               
          return tf.add_n(tf.get_collection('losses'), name='total_loss')

     def get_placeholders(self):

          image_multigpu_placeholders = []
          label_multigpu_placeholders = []

          for i in range(FLAGS.num_gpus):
               image_multigpu_placeholders.append(tf.placeholder(tf.float32, shape=(FLAGS.batch_size, PRIOR_SIZE, PRIOR_SIZE, 2)))
               label_multigpu_placeholders.append(tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 3)))
          
          return image_multigpu_placeholders, label_multigpu_placeholders

     def tower_loss(self, images, label, scope):

          """Calculate the total loss on a single tower running the model.
          Args:
          scope: unique prefix string identifying the tower, e.g. 'tower_0'
          images: placeholder labels for this tower
          label: placeholder labels for this tower
          Returns:
          Tensor of shape [] containing the total loss for a batch of data
          """

          print images
          output = self.inference(images)
          
          total_loss = self.loss(output, label)

          # using self.output will give last gpu outputs only
          with tf.variable_scope("outputs"):
               tf.summary.histogram("x_outputs", output[:,0])
               tf.summary.histogram("y_outputs", output[:,1])
               tf.summary.histogram("theta_outputs", output[:,2])
          with tf.variable_scope("inputs"):
               tf.summary.histogram("x_inputs", label[:,0])
               tf.summary.histogram("y_inputs", label[:,1])
               tf.summary.histogram("theta_inputs", label[:,2])

          return total_loss

     def average_gradients(self, tower_grads):

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
                    grad = tf.concat(axis=0, values=grads)
                    grad = tf.reduce_mean(grad, 0)

                    # Keep in mind that the Variables are redundant because they are shared
                    # across towers. So .. we will just return the first tower's pointer to
                    # the Variable.
                    v = grad_and_vars[0][1]
                    grad_and_var = (grad, v)
                    average_grads.append(grad_and_var)
                    
          return average_grads


     def train(self):

          with tf.Graph().as_default(), tf.device('/cpu:0'):
               global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
               num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)
               decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
         
               # Decay the learning rate exponentially based on the number of steps.
               lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

               opt = tf.train.GradientDescentOptimizer(INITIAL_LEARNING_RATE)
         
               # Calculate gradients for each model tower
               tower_grads = []

               images_multigpu, labels_multigpu = self.get_placeholders()
               print len(images_multigpu)
               #split_images = tf.spliti(images, FLAGS.num_gpus, axis=0)
               #split_labelss = tf.split(labels, FLAGS.num_gpus, axis=0)
               
               with tf.variable_scope(tf.get_variable_scope()):
                    for i in xrange(FLAGS.num_gpus):
                         with tf.device('/gpu:%d' % i):
                              with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                                   # Calculate the loss for one tower of the model. This function
                                   # constructs the entire model but shares the variables across
                                   # all towers.
                                   
                                   loss = self.tower_loss(images_multigpu[i], labels_multigpu[i], scope)
                                   
                                   # Reuse variables for the next tower.
                                   tf.get_variable_scope().reuse_variables()

                                   # Retain the summaries from the final tower.
                                   summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                                   # Retain the Batch Normalization updates operations only from the
                                   # final tower. Ideally, we should grab the updates from all towers
                                   # but these stats accumulate extremely fast so we can ignore the
                                   # other stats from the other towers without significant detriment.
                                   batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                                   
                                   # Calculate the gradients for the batch of data on this CIFAR tower.
                                   grads = opt.compute_gradients(loss)

                                   # Keep track of the gradients across all towers.
                                   tower_grads.append(grads)

               # We must calculate the mean of each gradient. Note that this is the
               # synchronization point across all towers.
               grads = self.average_gradients(tower_grads)

               # Add a summary to track the learning rate.
               summaries.append(tf.summary.scalar('learning_rate', lr))

               #self.return_train_op(total_loss,global_step)

	       
               # Add histograms for gradients.
               for grad, var in grads:
                    if grad is not None:
                         summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
	       
               # Apply the gradients to adjust the shared variables.
               apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)    

               # Add histograms for trainable variables.
               for var in tf.trainable_variables():
                   summaries.append(tf.summary.histogram(var.op.name, var))

               # Track the moving averages of all trainable variables
               variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
               variables_averages_op = variable_averages.apply(tf.trainable_variables())

               batchnorm_updates_op = tf.group(*batchnorm_updates)

               # Group all updates to into a single train op.
               self.train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

               saver = tf.train.Saver(tf.global_variables())
               # Build the summary operation from the last tower summaries.
               summary_op = tf.summary.merge(summaries)

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

               #print 'testing'
               #sys.exit()
               
               for step in range(FLAGS.max_steps):
                    start_time = time.time()
                    batch_indices = self.minibatch_indices(step)

                    # print batch_indices
                    minibatch = self.load_minibatch(batch_indices)
                    # print minibatch[0].shape
                    # print minibatch[1].shape

                    feed_dict = {}
                    for gpu_index in range(FLAGS.num_gpus):
                         feed_dict[images_multigpu[gpu_index] ] = minibatch[0][i]
                         feed_dict[labels_multigpu[gpu_index] ] = minibatch[1][i]
                      
                    #feed_dict = { images : minibatch[0], label:minibatch[1]}
        
                    if step % FLAGS.save_pred_every == 0:
                         loss_value,_, summary = sess.run([loss, self.train_op , summary_op], feed_dict=feed_dict)
                         summary_writer.add_summary(summary, step)
                         # save(saver, sess, args.snapshot_dir, step)
                    else:
                         loss_value, _ = sess.run([loss, self.train_op], feed_dict=feed_dict)
             
                    duration = time.time() - start_time
                    if step % FLAGS.print_pred_every == 0:
                         print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

     def minibatch_indices(self, step):

          counter_start = FLAGS.num_gpus * step * FLAGS.batch_size % self.num_items
          counter_end  = FLAGS.num_gpus * (step+1) * FLAGS.batch_size % self.num_items

          if counter_end > counter_start:
               batch_indices = self.indices[counter_start : counter_end    ]
          else:
               batch_indices = np.zeros(FLAGS.num_gpus * FLAGS.batch_size, dtype = np.int)
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
          self.truth_label = b['output'][:num_items]
          self.truth_label = self.truth_label.astype(np.float32, copy=False)

          self.truth_label[:,0] =self.truth_label[:,0]/100 
          self.truth_label[:,1] =self.truth_label[:,1]/100 
          self.truth_label[:,2] =self.truth_label[:,2]/100 

          self.num_items = num_items
          print  "local_map shape " , self.local_map.shape
          print "prior_map shape " , self.prior_map.shape
          print "outputs shape " , self.truth_label.shape
          print "Num of examples(augmented)" , self.num_items

     def load_minibatch(self, indices):

          images = np.zeros( (FLAGS.num_gpus, indices.size / FLAGS.num_gpus , PRIOR_SIZE, PRIOR_SIZE, 2 )  )
          output = np.zeros( (FLAGS.num_gpus, indices.size / FLAGS.num_gpus, 3 )  )
          padding = (PRIOR_SIZE - LOCAL_SIZE) / 2
          batch_per_gpu = indices.shape[0] / FLAGS.num_gpus
          
          for gpu_index in range(FLAGS.num_gpus):
               for counter in range(batch_per_gpu):
                    baseIndex = batch_per_gpu * gpu_index
                    prior_map_index = indices[baseIndex+counter]
                    local_map_index = prior_map_index / 20
                    images[gpu_index,counter,:,:,0] = np.pad(np.reshape(self.local_map[local_map_index],(LOCAL_SIZE, LOCAL_SIZE) ),[[padding, padding], [padding, padding]],'constant', constant_values=(0, 0))
                    images[gpu_index,counter,:,:,1] = np.reshape(self.prior_map[prior_map_index],(PRIOR_SIZE,PRIOR_SIZE) )
                    output[gpu_index,counter,:] = self.truth_label[prior_map_index]

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
