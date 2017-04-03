import tensorflow as tf
import data_input as data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("summary_dir", "/media/data_raid/nikhil/test_summary/run_1",
                           """Dir path for testing summary""")

def test_input():

    with tf.Graph().as_default():
        # Input images and labels

        images, outputs = data.inputs(is_train=True, batch_size=10, num_epochs=500)

        with tf.name_scope("summary_logs") as scope:
            tf.summary.image("local", tf.expand_dims(images[:,:,:,1],-1))
            tf.summary.image("prior", tf.expand_dims(images[:,:,:,0],-1))

            tf.summary.histogram("x_outputs", outputs[:,0])
            tf.summary.histogram("y_outputs", outputs[:,1])
            tf.summary.histogram("thet_outputs", outputs[:,2])

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
        test_writer = tf.summary.FileWriter(FLAGS.summary_dir + "/test", sess.graph)

        summary_result = sess.run(summary_op)
        test_writer.add_summary(summary_result, 1)

        # When done, ask the threads to stop.
        coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        
def main(argv=None):
    if tf.gfile.Exists(FLAGS.summary_dir):
        tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)
    print ("Summary Directory is %s" % (FLAGS.summary_dir))
    test_input()

if __name__ == '__main__':
    tf.app.run()
