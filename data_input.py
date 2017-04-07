import tensorflow as tf
import os
import glob

FLAGS = None

#TRAIN_FILE = "train.tfrecords"
#VALIDATION_FILE = "validation.tfrecords"
DIRECTORY = "/media/data_raid/dnn_localization/localization_dataset/tf_records/"

PRIOR_SIZE = 500
LOCAL_SIZE = 200

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features = {
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'output_raw': tf.FixedLenFeature([], tf.string),
                                           #'height': tf.FixedLenFeature([], tf.int64),
                                           #'depth': tf.FixedLenFeature([], tf.int64),
                                           #'width': tf.FixedLenFeature([], tf.int64),
                                        })

    image = tf.decode_raw(features['image_raw'], tf.float64)
    output = tf.decode_raw(features['output_raw'], tf.int32)

    image = tf.cast(image, tf.float32)
    output = tf.cast(output, tf.float32)
    output = output / 100
    #print image.shape
    #image = tf.reshape(image, (PRIOR_SIZE, PRIOR_SIZE, 2))
    #image.set_shape([PRIOR_SIZE,PRIOR_SIZE,2])

    image.set_shape([PRIOR_SIZE*PRIOR_SIZE*2])
    output.set_shape([3])
            
    return image, output

def inputs(is_train, batch_size, num_epochs):

    """Reads input data num_epochs times.
    Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, PRIOR_SIZE, PRIOR_SIZE, 2].
    * output is a float tensor with shape [batch_size, 3] with [x, y, theta].
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    
    if not num_epochs: num_epochs = None
    dirPath = os.path.join(DIRECTORY, "train_dir/*.tfrecords" if is_train else "validation_dir/*.tfrecords")

    filename_list = glob.glob(dirPath)
    print ("Number of files: %d") % len(filename_list)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer( filename_list, num_epochs=num_epochs, shuffle=True)
        
        image, output = read_and_decode(filename_queue)
        
        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a Random Shuffle Queue.)
        # We run this in two threads to avoid being a bottleneck

        images, outputs = tf.train.shuffle_batch(
            [image, output], batch_size=batch_size, num_threads=2,
            capacity=500+2*batch_size,
            # Ensures a minimum amount of shuffling of examples
            min_after_dequeue=200,
            )

        #print "After Shuffle Size: " + str(images.shape)
        images = tf.reshape(images, (batch_size, PRIOR_SIZE, PRIOR_SIZE, 2))

        #print images.shape
        #print outputs.shape
        
        return images, outputs
            
