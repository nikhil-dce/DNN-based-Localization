import tensorflow as tf
from tensorflow.python.client import timeline

import data_input as data
from model import RegressionModel
import model

import numpy as np
import os
import sys
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
NUMBER_GPU = 2
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('root_dir', '/media/data_raid/nikhil/events_summary',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 200000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', NUMBER_GPU,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('batch_size', 20,
                           """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('save_pred_every', 20,
                           """Save summary frequency""")
tf.app.flags.DEFINE_integer('print_pred_every', 20,
                           """Print loss every steps""")
tf.app.flags.DEFINE_integer('save_valid_every', 500,
                            """Save Validation summary""")
tf.app.flags.DEFINE_integer('save_checkpoint_every', 1000,
                            """Save checkpoint/snapshot""")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 30
# Constants describing the training process.

NUM_EPOCHS_PER_DECAY = 350.0    # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1# Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-5   # Initial learning rate.
INITIAL_LEARNING_RATE_ADAM = 1e-5   # Initial learning rate.

TOWER_NAME = 'tower'


def test_input():

    with tf.Graph().as_default():
        # Input images and labels

        images, outputs = data.inputs(is_train=True, batch_size=5, num_epochs=500)

        with tf.name_scope("summary_logs") as scope:
            tf.summary.image("local", tf.expand_dims(images[:,:,:,1],-1))
            tf.summary.image("prior", tf.expand_dims(images[:,:,:,0],-1))

            tf.summary.histogram("x_outputs", outputs[:,0])
            tf.summary.histogram("y_outputs", outputs[:,1])
            tf.summary.histogram("theta_outputs", outputs[:,2])

            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            summary_op = tf.summary.merge(summaries)



        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess = tf.Session()

        sess.run(init_op)
        
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print "Session Run"
        test_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        summary_result = sess.run(summary_op)
        test_writer.add_summary(summary_result, 1)

        # When done, ask the threads to stop.
        coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

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
    with tf.name_scope("average_gradients"):
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
        
def train():

    with tf.Graph().as_default(), tf.device('/cpu:0'):
    
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        #num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)
        #decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        
        # Decay the learning rate exponentially based on the number of steps.
        #lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
        #                            global_step,
        #                            decay_steps,
        #                            LEARNING_RATE_DECAY_FACTOR,
        #                            staircase=True)

        #opt = tf.train.GradientDescentOptimizer(INITIAL_LEARNING_RATE)
        opt = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE_ADAM)

        # Calculate gradients for each model tower
        tower_grads = []
        
        regressionModel = RegressionModel() 

        phaseTrain = True

        # Get input based on phase for each tower
        images, labels = data.inputs(is_train=phaseTrain, batch_size=(FLAGS.batch_size*FLAGS.num_gpus), num_epochs=None)

        split_images = tf.split(images, FLAGS.num_gpus, 0)
        split_labels = tf.split(labels, FLAGS.num_gpus, 0)
        
        with tf.variable_scope(tf.get_variable_scope()) as main_scope:
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:

                        #phaseTrain = True
                        # Get input based on phase for each tower
                        #images, labels = data.inputs(is_train=phaseTrain, batch_size=FLAGS.batch_size, num_epochs=None)
                        
                        
                        # Calculate the loss for one tower of the model. This function
                        # constructs the entire model but shares the variables across
                        # all towers.

                        # Calculating tower loss
                        output = regressionModel.inference(split_images[i], phaseTrain)
                        loss = regressionModel.loss(output, split_labels[i]) 
                                                                                   
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

        #print(tf.get_default_graph().as_graph_def())
        #sys.exit(0)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        # summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
	       
        # Apply the gradients to adjust the shared variables.
        with tf.name_scope("gradient_apply"):
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)    

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables
        with tf.name_scope("exp_moving_average"):
            variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

        batchnorm_updates_op = tf.group(*batchnorm_updates)

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

        queue_summary = tf.get_collection(tf.GraphKeys.SUMMARIES, 'input')
        summaries.append(queue_summary)
        
        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        """
        # Validation Code
        vx_error_sqr = []
        vx_error_abs = []
        vy_error_sqr = []
        vy_error_abs = []
        vtheta_error_sqr = []
        vtheta_error_abs = []
        vweighted_error_sqr = []
        vweighted_error_abs = []

        with tf.variable_scope(main_scope, reuse = True): # To allow variable reuse from main_scope 
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('Validation_%s_%d' % (TOWER_NAME, i)) as scope: # This will only change name scopes for operations (Vars still the same)
                        
                        phaseValidation = False

                        # Get input based for validation
                        images, labels = data.inputs(is_train=phaseValidation, batch_size=FLAGS.batch_size, num_epochs=None)
                        
                        # Calculate the loss for one tower of the model. This function
                        # constructs the entire model but shares the variables across
                        # all towers.

                        # Calculating tower loss
                        output = regressionModel.inference(images, phaseValidation)
                        batch_loss_array = regressionModel.batch_loss(output, labels) 
                        
                        # Accumulate error for validation loss
                        vweighted_error_sqr.append(batch_loss_array[0])
                        vweighted_error_abs.append(batch_loss_array[1])
                        vx_error_sqr.append(batch_loss_array[2])
                        vx_error_abs.append(batch_loss_array[3])
                        vy_error_sqr.append(batch_loss_array[4])
                        vy_error_abs.append(batch_loss_array[5])
                        vtheta_error_sqr.append(batch_loss_array[6])
                        vtheta_error_abs.append(batch_loss_array[7])

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                                                
        with tf.name_scope("validation_accumulate"):
            vx_total_error_sqr = tf.add_n(vx_error_sqr)
            vy_total_error_sqr = tf.add_n(vy_error_sqr)
            vtheta_total_error_sqr = tf.add_n(vtheta_error_sqr)
            vweighted_total_error_sqr = tf.add_n(vweighted_error_sqr)
            vx_total_error_abs = tf.add_n(vx_error_abs)
            vy_total_error_abs = tf.add_n(vy_error_abs)
            vtheta_total_error_abs = tf.add_n(vtheta_error_abs)
            vweighted_total_error_abs = tf.add_n(vweighted_error_abs)

        # Validation Summaries from placeholders
        with tf.name_scope("validation_summary"):

            x_mse = tf.placeholder(tf.float32, name="x_mse")
            y_mse = tf.placeholder(tf.float32, name="y_mse")
            theta_mse = tf.placeholder(tf.float32, name="theta_mse")
            weighted_mse = tf.placeholder(tf.float32, name="weighted_mse")
            
            x_mae = tf.placeholder(tf.float32, name="x_mae")
            y_mae = tf.placeholder(tf.float32, name="y_mae")
            theta_mae = tf.placeholder(tf.float32, name="theta_mae")
            weighted_mae = tf.placeholder(tf.float32, name="weighted_mae")

            validation_summaries = []
            validation_summaries.append(tf.summary.scalar('x_mse', x_mse))
            validation_summaries.append(tf.summary.scalar('y_mse', y_mse))
            validation_summaries.append(tf.summary.scalar('theta_mse', theta_mse))
            validation_summaries.append(tf.summary.scalar('loss_mse', weighted_mse))

            validation_summaries.append(tf.summary.scalar('x_mae', x_mae))
            validation_summaries.append(tf.summary.scalar('y_mae', y_mae))
            validation_summaries.append(tf.summary.scalar('theta_mae', theta_mae))
            validation_summaries.append(tf.summary.scalar('loss_mae', weighted_mae))

            validation_summary_op = tf.summary.merge(validation_summaries)
        """
        # Setting up session and initialization stuf

               
        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        
        sess.run(init)
        
        print("EVERYTHING IS INITIALIZED")

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        meta_graph_def = tf.train.export_meta_graph(filename=FLAGS.train_dir+'/my-model.meta')
        
        print("GRAPH IS  SAVED")
                
        # Training loop 
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        
        time.sleep(50)
        for step in range(FLAGS.max_steps):
            start_time = time.time()

            
#            queue1 = sess.run("input/shuffle_batch/random_shuffle_queue_Size:0")
#           queue2 = sess.run("tower_1/input/shuffle_batch/random_shuffle_queue_Size")

#            print ("Queue1 Size: %d and Queue2 Size: %d") % (queue1, 0)
            
            feed_dict = {}

            # Timeline
            #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #run_metadata = tf.RunMetadata()
            
            if step % FLAGS.save_pred_every == 0:
                loss_value, _ , summary = sess.run([loss, train_op, summary_op])#, options = run_options, run_metadata = run_metadata)
                summary_writer.add_summary(summary, step)
                # save(saver, sess, args.snapshot_dir, step)
            else:
                loss_value, _ = sess.run([loss, train_op])#, options = run_options, run_metadata = run_metadata)

            """
            # Create Timeline object, and write to json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open((FLAGS.train_dir + '/timeline_%d.json') % step , 'w') as f:
                f.write(ctf)
            """
            """
            if step % FLAGS.save_valid_every == 0:
                            
                items = FLAGS.test_items
                validation_sessions = items / (FLAGS.num_gpus * FLAGS.batch_size)

                output_tensors = []
                output_tensors.append(vweighted_total_error_sqr)                
                output_tensors.append(vx_total_error_sqr)
                output_tensors.append(vy_total_error_sqr)
                output_tensors.append(vtheta_total_error_sqr)
                output_tensors.append(vweighted_total_error_abs)
                output_tensors.append(vx_total_error_abs)
                output_tensors.append(vy_total_error_abs)
                output_tensors.append(vtheta_total_error_abs)
                                
                vloss_arr = np.empty((0,8))

                print ("Validation: Items being used: %d and session counter: %d") % (items, validation_sessions)
                                
                for session_counter in range(validation_sessions):
                    validation_error_array = sess.run(output_tensors, feed_dict=feed_dict)
                    validation_error_array = np.array(validation_error_array).reshape(1,8)
                    vloss_arr = np.append(vloss_arr, validation_error_array, axis=0)
                
                final_vloss = vloss_arr.sum(axis=0)
                final_vloss = final_vloss / items

                feed_dict[weighted_mse] = final_vloss[0]
                feed_dict[x_mse] = final_vloss[1]
                feed_dict[y_mse] = final_vloss[2]
                feed_dict[theta_mse] = final_vloss[3]
                
                feed_dict[weighted_mae] = final_vloss[4]
                feed_dict[x_mae] = final_vloss[5]
                feed_dict[y_mae] = final_vloss[6]
                feed_dict[theta_mae] = final_vloss[7]
                                
                print 'Validation Summary MSE {:.3f}, \t MAE {:.3f}'.format(final_vloss[0], final_vloss[4])
                
                validation_summary = sess.run(validation_summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(validation_summary, step)
            """         
            duration = time.time() - start_time
            # Now duration also includes the validation time...

            if step % FLAGS.print_pred_every == 0:
                print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

            # Save the model checkpoint periodically.
            if step % FLAGS.save_checkpoint_every == 0 or (step + 1) == FLAGS.max_steps:
                print 'Saving model Checkpoint...'
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
        
        coord.request_stop()
        
        # Wait for threads to finish
        coord.join(threads)
        sess.close()



def main(argv=None):

    RUN_DIR = "run_14"
    TEST_ITEMS = 1000
    FLAGS.test_items = (TEST_ITEMS / (FLAGS.num_gpus * FLAGS.batch_size)) * FLAGS.num_gpus * FLAGS.batch_size
    FLAGS.train_dir = os.path.join(FLAGS.root_dir, RUN_DIR) 

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    print ("TRAIN Directory is %s" % (FLAGS.train_dir))

    #test_input()
    train()

if __name__ == '__main__':
    tf.app.run()
