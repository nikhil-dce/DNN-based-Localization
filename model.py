import tensorflow as tf
import numpy as np
import time
import sys
import re
import os

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

class batch_norm(object):
     
     #def __init__(self, epsilon=1e-5, momentum = 0.99, name="batch_norm"):
     def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
          with tf.variable_scope(name), tf.device('/cpu:0'):
               self.epsilon  = epsilon
               self.momentum = momentum
               self.name = name

     def __call__(self, x, phase):
          return tf.contrib.layers.batch_norm(x,
                                              decay=self.momentum, 
                                              epsilon=self.epsilon,
                                              scale=True,
                                              center=True, 
                                              is_training=phase,
                                              scope=self.name)

class RegressionModel:

     def __init__(self,  trainable=True):
          self.trainable = trainable
          self.weights = [1,1,1]

     def twin_build(self, prior, local, phase):
          
          tf.summary.image("prior", prior)
          tf.summary.image("local", local)

          with tf.name_scope("twin_1"):
               with tf.variable_scope("twin_shared") as twin_shared:
                    conv1_prior = self.conv_layer(prior, 1, 32, "conv1", filter_size=3, padding="SAME")

               conv1_prior = self.max_pool(conv1_prior, "max_pool_prior_1")
               conv1_prior = batch_norm(name='bn_conv1_prior')(conv1_prior, phase)

               with tf.variable_scope(twin_shared):
                    conv2_prior = self.conv_layer(conv1_prior, 32, 64, "conv2", filter_size=3, padding="SAME")

               conv2_prior = self.max_pool(conv2_prior, "max_pool_prior_2")
               conv2_prior = batch_norm(name='bn_conv2_prior')(conv2_prior, phase)

               with tf.variable_scope(twin_shared):
                    conv3_prior = self.conv_layer(conv2_prior, 64, 128, "conv3", filter_size=3, padding="SAME")

               conv3_prior = self.max_pool(conv3_prior, "max_pool_prior_3")
               conv3_prior = batch_norm(name='bn_conv3_prior')(conv3_prior, phase)

               with tf.variable_scope(twin_shared):
                    conv4_prior = self.conv_layer(conv3_prior, 128, 256, "conv4", filter_size=3, padding="SAME")

               conv4_prior = self.max_pool(conv4_prior, "max_pool_prior_4")
               conv4_prior = batch_norm(name='bn_conv4_prior')(conv4_prior, phase)
               
               fc_prior = self.fc_layer(conv4_prior, 32*32*256, 1024,"fc_prior")
          
               if phase:
                    fc_prior_dropout = self.dropout(fc_prior, 0.5, "dropout_fc_prior")
               else:
                    fc_prior_dropout = self.dropout(fc_prior, 1, "dropout_fc_prior")


          with tf.name_scope("twin_2"):
               with tf.variable_scope(twin_shared, reuse=True):
                    conv1_local = self.conv_layer(local, 1, 32, "conv1", filter_size=3, padding="SAME")
          
               conv1_local = self.max_pool(conv1_local, "max_pool_local_1")
               conv1_local = batch_norm(name='bn_conv1_local')(conv1_local, phase)

               with tf.variable_scope(twin_shared, reuse=True):
                    conv2_local = self.conv_layer(conv1_local, 32, 64, "conv2", filter_size=3, padding="SAME")
         
               conv2_local = self.max_pool(conv2_local, "max_pool_local_2")
               conv2_local = batch_norm(name='bn_conv2_local')(conv2_local, phase)

               with tf.variable_scope(twin_shared, reuse=True):
                    conv3_local = self.conv_layer(conv2_local, 64, 128, "conv3", filter_size=3, padding="SAME")

               conv3_local = self.max_pool(conv3_local, "max_pool_local_3")
               conv3_local = batch_norm(name='bn_conv3_local')(conv3_local, phase)

               with tf.variable_scope(twin_shared, reuse=True):
                    conv4_local = self.conv_layer(conv3_local, 128, 256, "conv4", filter_size=3, padding="SAME")

               conv4_local = self.max_pool(conv4_local, "max_pool_local_4")
               conv4_local = batch_norm(name='bn_conv4_local')(conv4_local, phase)

               fc_local = self.fc_layer(conv4_local, 13*13*256, 256,"fc_local")
          
               if phase:
                    fc_local_dropout = self.dropout(fc_local, 0.5, "dropout_fc_local")
               else:
                    fc_local_dropout = self.dropout(fc_local, 1, "dropout_fc_local")

          # Merged FC
          merged_feature = tf.concat([fc_prior_dropout, fc_local_dropout], 1)
          print merged_feature.get_shape()
          fc_prior_local = self.fc_layer(merged_feature, 1280, 256, "fc_prior_local")

          if phase:
               fc_prior_local_dropout = self.dropout(fc_prior_local, 0.5, "dropout_fc_prior_local")
          else:
               fc_prior_local_dropout = self.dropout(fc_prior_local, 1, "dropout_fc_prior_local")
          
          fc_output = self.fc_layer(fc_prior_local_dropout, 256, 3, "fc_output")
          
          self.output = fc_output
          
     def build(self, images, phase):

          """
          Define model architecture
          """
          self.images = images
          tf.summary.image("prior",tf.expand_dims(images[:,:,:,0],-1))
          tf.summary.image("local",tf.expand_dims(images[:,:,:,1],-1))
          
          # Will make network deeper and shallower following the VGG guidelines
          # While applying max pool increase the following layers filter depth
          # Only use 3X3 filters
          # Atleast 2 FC with dropout
          # Max pool not after every conv
          
          self.conv1 = self.conv_layer(images, 2, 16, "conv1",filter_size=3,padding="SAME")
          self.conv1 = self.max_pool(self.conv1, "max_pool_1")
          self.conv1 = batch_norm(name='bn_conv1')(self.conv1, phase)

          self.conv2 = self.conv_layer(self.conv1, 16, 32, "conv2", filter_size=3, padding="SAME")
          self.conv2 = self.max_pool(self.conv2,"max_pool_2")
          self.conv2 = batch_norm(name='bn_conv2')(self.conv2, phase)

          self.conv3 = self.conv_layer(self.conv2, 32, 64, "conv3", filter_size=3, padding="SAME")
          self.conv3 = self.max_pool(self.conv3,"max_pool_3")
          self.conv3 = batch_norm(name='bn_conv3')(self.conv3, phase)

          self.conv4 = self.conv_layer(self.conv3, 64, 128, "conv4", filter_size=3, padding="SAME")
          self.conv4 = self.max_pool(self.conv4, "max_pool_4")
          self.conv4 = batch_norm(name='bn_conv4')(self.conv4, phase)

          self.conv5 = self.conv_layer(self.conv4, 128, 256, "conv5", filter_size=3, padding="SAME")
          self.conv5 = self.max_pool(self.conv5,"max_pool_5")
          self.conv5 = batch_norm(name='bn_conv5')(self.conv5, phase)

          """
          
          self.conv6 = self.conv_layer(self.conv5, 64, 64, "conv6", filter_size=3, padding="SAME")
          self.conv6 = self.max_pool(self.conv6, "max_pool_6")
          self.conv6 = batch_norm(name='bn_conv6')(self.conv6, phase)

          self.conv7 = self.conv_layer(self.conv6, 64, 128, "conv7", filter_size=3, padding="SAME")
          self.conv7 = self.max_pool(self.conv7,"max_pool_7")
          self.conv7 = batch_norm(name='bn_conv7')(self.conv7, phase)

          self.conv8 = self.conv_layer(self.conv7, 128, 128, "conv8", filter_size=3, padding="SAME")
          self.conv8 = self.max_pool(self.conv8, "max_pool_8")
          self.conv8 = batch_norm(name='bn_conv8')(self.conv8, phase)
          
          """
          print "FC input: " + str(self.conv5.get_shape())

          self.fc_1 = self.fc_layer(self.conv5,16*16*256, 256,"fc_1")
          #self.fc_1 = self.fc_layer(self.conv8,2*2*128, 256,"fc_1")

          if phase:
               self.fc_1_dropout = self.dropout(self.fc_1, 0.5, "dropout_1")
          else:
               self.fc_1_dropout = self.dropout(self.fc_1, 1, "dropout_1")

          self.fc_2 = self.fc_layer(self.fc_1_dropout, 256, 128, "fc_2")

          if phase:
               self.fc_2_dropout = self.dropout(self.fc_2, 0.5, "dropout_2")
          else:
               self.fc_2_dropout = self.dropout(self.fc_2, 1, "dropout_2")

          self.fc_3 = self.fc_layer(self.fc_2_dropout, 128, 3, "fc_3")
               

          self.output = self.fc_3

     def twin_inference(self, prior, local, phase):
          self.twin_build(prior, local, phase)
          print "______________________________"
          print "Twin Build"
          return self.output

     def inference(self,images,phase):
          self.build(images,phase)
          print "______________________________"
          print "Network Built"
          return self.output

     def loss(self, output, label):

          self.label = label
          # squared_error = tf.losses.mean_squared_error(output , label)
          weights = self.weights
                    
          x_error_abs = tf.abs(output[:,0] - label[:,0])
          y_error_abs = tf.abs(output[:,1] - label[:,1])
          theta_error_abs = tf.minimum( tf.abs(output[:,2] - label[:,2]), tf.abs(output[:,2] + 3.6 - label[:,2]) )
          theta_error_abs = tf.minimum( theta_error_abs, tf.abs(output[:,2] - 3.6 - label[:,2]) ) 

          xerror_batch = tf.square(x_error_abs)
          yerror_batch = tf.square(y_error_abs)
          #theta_error1 = tf.square(theta_error_abs) 

          xy_l2 = tf.sqrt(tf.add(xerror_batch, yerror_batch))

          """
          x_mse = tf.reduce_mean(xerror_batch)
          y_mse = tf.reduce_mean(yerror_batch)
          theta_mse = tf.reduce_mean(theta_error1) 
          weighted_mse = weights[0]*x_mse + weights[1]*y_mse + weights[2]*theta_mse
          """
          
          #x_mae = tf.reduce_mean(x_error_abs)
          #y_mae = tf.reduce_mean(y_error_abs)
          xy_mae = tf.reduce_mean(xy_l2)
          theta_mae = tf.reduce_mean(theta_error_abs)
          weightedLoss = weights[0]*xy_mae + weights[1]*theta_mae
          #weighted_mae = weights[0]*x_mae + weights[1]*y_mae + weights[2]*theta_mae

          with tf.variable_scope("output_loss"):

               #tf.summary.scalar("x_mse", x_mse)
               #tf.summary.scalar("y_mse", y_mse)
               #tf.summary.scalar("theta_mse", theta_mse)
               #tf.summary.scalar("loss_mse", weighted_mse)
               
               tf.summary.scalar("xy_mae", xy_mae)
               #tf.summary.scalar("y_mae", y_mae) 
               tf.summary.scalar("theta_mae", theta_mae)
               tf.summary.scalar("weighted_mae", weightedLoss)
               
               # beta = 0.001
               # lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars
               # if 'bias' not in v.name ]) * beta
               # Add L2 loss later in the future

          with tf.variable_scope("outputs"):
               tf.summary.histogram("x_outputs", output[:,0])
               tf.summary.histogram("y_outputs", output[:,1])
               tf.summary.histogram("theta_outputs", output[:,2])
          with tf.variable_scope("inputs"):
               tf.summary.histogram("x_inputs", label[:,0])
               tf.summary.histogram("y_inputs", label[:,1])
               tf.summary.histogram("theta_inputs", label[:,2])
     
          return weightedLoss

     def batch_loss(self, output, label):

          self.label = label
          # squared_error = tf.losses.mean_squared_error(output , label)
          weights = self.weights

          # Calculate both abs and sqr for MAE and RMS respec
          x_error_abs = tf.abs(output[:,0] - label[:,0])
          y_error_abs = tf.abs(output[:,1] - label[:,1])
          theta_error_abs = tf.minimum( tf.abs(output[:,2] - label[:,2]), tf.abs(output[:,2] + 3.6 - label[:,2]) )
          theta_error_abs = tf.minimum( theta_error_abs, tf.abs(output[:,2] - 3.6 - label[:,2]) ) 
          
          xy_batch_error = tf.sqrt( tf.add(tf.square(x_error_abs), tf.square(y_error_abs) ))
          xy_batch_total_error = tf.reduce_sum(xy_batch_error)

          theta_batch_total_error = tf.reduce_sum(theta_error_abs)

          weightedLoss = weights[0]*xy_batch_total_error + weights[1]*theta_batch_total_error

          """
          # will return total batch error -> needed in validation case
          x_batch_error_sqr = tf.reduce_sum( tf.square(x_error_abs) )
          y_batch_error_sqr = tf.reduce_sum( tf.square(y_error_abs) )
          theta_batch_error_sqr = tf.reduce_sum( tf.square(theta_error_abs) )
          weighted_error_sqr = weights[0]*x_batch_error_sqr + weights[1]*y_batch_error_sqr + weights[2]*theta_batch_error_sqr

          x_batch_error_abs = tf.reduce_sum(x_error_abs)
          y_batch_error_abs = tf.reduce_sum(y_error_abs)
          theta_batch_error_abs = tf.reduce_sum(theta_error_abs)
          weighted_error_abs = weights[0]*x_batch_error_abs + weights[1]*y_batch_error_abs + weights[2]*theta_batch_error_abs
          """

          #return (weighted_error_sqr, weighted_error_abs, x_batch_error_sqr, x_batch_error_abs, y_batch_error_sqr, y_batch_error_abs, theta_batch_error_sqr, theta_batch_error_abs)
          return (weightedLoss, xy_batch_total_error, theta_batch_total_error)

     def dropout(self, bottom, keep_prob, name):
          return tf.nn.dropout(bottom, keep_prob, name=name)

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

