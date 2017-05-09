
import os
import sys
import argparse
import h5py
import numpy as np
import tensorflow as tf

PRIOR_SIZE = 500
LOCAL_SIZE = 200
LOCAL_PRIOR_MAPPING = 50

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(filename, prior, local, output):
    # write code to save data in dirPath
    examplesToSave = prior.shape[0]
    print 'Saving %d examples in %s' % (examplesToSave, filename)

    rows = PRIOR_SIZE
    cols = PRIOR_SIZE
    depth = 2
    padding = (PRIOR_SIZE - LOCAL_SIZE)/2
    
    fileToSave = os.path.join(FLAGS.directory, filename + '.tfrecords')
    print ('Writing', fileToSave)

    writer = tf.python_io.TFRecordWriter(fileToSave)
    for index in range(examplesToSave):

        prior_input = 255 * np.reshape(prior[index], (PRIOR_SIZE, PRIOR_SIZE))
        local_input = 255 * np.reshape(local[index/LOCAL_PRIOR_MAPPING], (LOCAL_SIZE, LOCAL_SIZE))
        out = output[index]

        prior_input = prior_input.astype(np.uint8)
        local_input = local_input.astype(np.uint8)

        prior_raw = prior_input.tostring()
        local_raw = local_input.tostring()
        out_raw = out.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'prior_raw': _bytes_feature(prior_raw),
            'local_raw': _bytes_feature(local_raw),
            'output_raw': _bytes_feature(out_raw)}) )
        writer.write(example.SerializeToString())

        """
        image_input = np.zeros((PRIOR_SIZE, PRIOR_SIZE, 2))

        image_input[:,:,0] = np.reshape(prior[index], (PRIOR_SIZE, PRIOR_SIZE))
        image_input[:,:,1] = np.pad(np.reshape(local[index/LOCAL_PRIOR_MAPPING], (LOCAL_SIZE, LOCAL_SIZE)), [[padding, padding], [padding,padding]], 'constant', constant_values=(0,0))
        out = output[index]

        image_raw = image_input.tostring()
        out_raw = out.tostring()
        
        example = tf.train.Example(features=tf.train.Features(feature={
            #'height': _int64_feature(rows),
            #'width' : _int64_feature(cols),
            #'depth' : _int64_feature(depth),
            'output_raw': _bytes_feature(out_raw),
            'image_raw' : _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
        """


    writer.close()

        
                                        
    
def main(argv=None):

    # get the data
    filedir = '/media/data_raid/dnn_localization/localization_dataset/'
    filename = 'annarbor_dataset_dnn_1.hdf5'
    fin = h5py.File(filedir+filename, 'r')

    # examples per file 
    EXAMPLES_PER_FILE = 5000
    
    totalExamples = fin['prior_map'].shape[0]
    testExamples = totalExamples - FLAGS.validation_size
    
    prior_map = fin['prior_map']
    local_map = fin['local_map']
    output = fin['output']

    
    # Decrease number of files
    # Before doing this test whether increasing the num_threads improve the performance
    
    examplesProcessed = 0
    fileCounter = 0
    while examplesProcessed < testExamples:
        
        examplesToSave = EXAMPLES_PER_FILE
        if examplesProcessed + examplesToSave > testExamples:
            examplesToSave = testExamples - examplesProcessed
        fileSuffix = "train_dir_3/%d" % fileCounter
        
        print "Saving %d to %d" % (examplesProcessed, examplesProcessed + examplesToSave)

        localBaseIndex = examplesProcessed / LOCAL_PRIOR_MAPPING
        localEndIndex = (examplesProcessed + examplesToSave) / LOCAL_PRIOR_MAPPING + 1

        convert_to(fileSuffix, prior_map[examplesProcessed:examplesProcessed+examplesToSave], local_map[localBaseIndex:localEndIndex], output[examplesProcessed:examplesProcessed + examplesToSave])
        examplesProcessed = examplesProcessed + examplesToSave
        fileCounter = fileCounter + 1
    
    examplesProcessed = testExamples
    fileCounter = 0
    while examplesProcessed < totalExamples:
        
        examplesToSave = EXAMPLES_PER_FILE
        if examplesProcessed + examplesToSave > totalExamples:
            examplesToSave = totalExamples - examplesProcessed
        fileSuffix = "validation_dir_3/%d" % fileCounter
        
        print "Saving %d to %d" % (examplesProcessed, examplesProcessed + examplesToSave)

        localBaseIndex = examplesProcessed / LOCAL_PRIOR_MAPPING
        localEndIndex = (examplesProcessed + examplesToSave) / LOCAL_PRIOR_MAPPING + 1

        convert_to(fileSuffix, prior_map[examplesProcessed:examplesProcessed+examplesToSave], local_map[localBaseIndex:localEndIndex], output[examplesProcessed:examplesProcessed + examplesToSave])
        examplesProcessed = examplesProcessed + examplesToSave
        fileCounter = fileCounter + 1
    
    """
    for counter in range(testExamples/EXAMPLES_PER_FILE):

        fileSuffix = "train_dir/%d" % counter

        baseIndex = counter * EXAMPLES_PER_FILE
        endIndex = baseIndex + EXAMPLES_PER_FILE

        print "BaseIndex: " + str(baseIndex)
        
        localBaseIndex = baseIndex / 50
        localEndIndex = endIndex / 50 + 1
        
        convert_to(fileSuffix, prior_map[baseIndex:endIndex], local_map[localBaseIndex:localEndIndex], output[baseIndex:endIndex])
    """
    """
    for counter in range(FLAGS.validation_size/EXAMPLES_PER_FILE):

        fileSuffix = "validation_dir/%d" % counter
        
        baseIndex = counter * EXAMPLES_PER_FILE + testExamples
        endIndex = baseIndex + EXAMPLES_PER_FILE

        print "BaseIndex: " + str(baseIndex)

        localBaseIndex = baseIndex / 50
        localEndIndex = endIndex / 50 + 1

        convert_to(fileSuffix, prior_map[baseIndex:endIndex], local_map[localBaseIndex:localEndIndex], output[baseIndex:endIndex])
            
    """               
    #convert_to(str(test_counter), prior_map[:testExamples], local_map[:testExamples/20], output[:testExamples])
    #convert_to('validation', prior_map[testExamples:totalExamples], local_map[testExamples/20:totalExamples/20], output[testExamples:totalExamples])
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='/media/data_raid/dnn_localization/localization_dataset/tf_records',
      help='Directory to write training tf records files'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
