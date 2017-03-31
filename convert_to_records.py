
import os
import sys
import argparse
import h5py
import numpy as np
import tensorflow as tf

PRIOR_SIZE = 500
LOCAL_SIZE = 200


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

        image_input = np.zeros((PRIOR_SIZE, PRIOR_SIZE, 2))

        image_input[:,:,0] = np.reshape(prior[index], (PRIOR_SIZE, PRIOR_SIZE))
        image_input[:,:,1] = np.pad(np.reshape(local[index/20], (LOCAL_SIZE, LOCAL_SIZE)), [[padding, padding], [padding,padding]], 'constant', constant_values=(0,0))
        out = output[index]

        image_raw = image_input.tostring()
        out_raw = out.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width' : _int64_feature(cols),
            'depth' : _int64_feature(depth),
            'output_raw': _bytes_feature(out_raw),
            'image_raw' : _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

        
                                        
    
def main(argv=None):
    # get the data
    filedir = '/media/data_raid/dnn_localization/localization_dataset/'
    filename = 'annarbor_dataset_dnn.hdf5'

    fin = h5py.File(filedir+filename, 'r')

    totalExamples = fin['prior_map'].shape[0]
    testExamples = totalExamples - FLAGS.validation_size

    prior_map = fin['prior_map']
    local_map = fin['local_map']
    output = fin['output']
    
    convert_to('train', prior_map[:testExamples], local_map[:testExamples/20], output[:testExamples])
    convert_to('validation', prior_map[testExamples:totalExamples], local_map[testExamples/20:totalExamples/20], output[testExamples:totalExamples])
    
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
