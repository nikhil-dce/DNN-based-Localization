import numpy as np
import tensorflow as tf

import time
import os
import math
import sys
from datetime import datetime

import data_input as data

import model
from model import RegressionModel

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/media/data_raid/nikhil/events_summary/eval_14',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/media/data_raid/nikhil/events_summary/run_14',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Batch Size""")

def eval_once(saver, summary_writer, batch_loss_array, variables_to_restore):

    print "Eval Once"
    with tf.Session() as sess:

        
        
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            # Restore from
            print "Checkpoint path: " + ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            # /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print ("No checkpoint exists")
            return

        #print (sess.run(variables_to_restore))
        #sys.exit(0)

        # Start the queue runners
        coord = tf.train.Coordinator()

                
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            total_loss = 0
            accumulate_loss = np.empty((0,8))
            while step < num_iter and not coord.should_stop():
                batch_loss_values = sess.run(batch_loss_array)
                batch_loss_values = np.array(batch_loss_values).reshape(1,8)
                accumulate_loss = np.append(accumulate_loss, batch_loss_values, axis=0)
                step += 1                        
            accumulate_loss = accumulate_loss.sum(axis=0)
            accumulate_loss /= (step*FLAGS.batch_size)

            print accumulate_loss
            print ('%s: Total Steps = %d' % (datetime.now(),  step))

            #summary = tf.Summary()
            #summary.ParseFromString(sess.run(summary_op))
            #summary_writer.add_summary(summary, global_step)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


                   
def evaluate():

    print "Evaluate"
    with tf.Graph().as_default() as g:

        r_model = RegressionModel()
        
        # Get validation input
        is_train = False
        
        images, label = data.inputs(is_train, FLAGS.batch_size, None)

        # Build a graph to output pose
        output = r_model.inference(images, is_train)
        batch_loss_array = r_model.batch_loss(output, label)
        
        variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, batch_loss_array, variables_to_restore)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
                

def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == '__main__':
    tf.app.run()
    
