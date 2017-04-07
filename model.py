import tensorflow as tf
import numpy as np
import time
import sys
import re
import os

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
                                            is_training=phase,
                                            scope=self.name)

class RegressionModel:

     def __init__(self,  trainable=True):
          self.trainable = trainable
          # these will change at the time of load_dataset
          self.train_items = 500
          self.test_items = 200
          self.weights = [1,1,1]


     def build(self, images, phase):

          """
          Define model architecture
          """
          self.images = images
          tf.summary.image("prior",tf.expand_dims(images[:,:,:,0],-1))
          tf.summary.image("local",tf.expand_dims(images[:,:,:,1],-1))
          
          self.conv1 =  self.conv_layer(images, 2, 16, "conv1",filter_size=5,padding="SAME")
          self.conv1 =  batch_norm(name='bn_conv1')(self.conv1, phase)
          self.conv1 = self.max_pool(self.conv1,"max_pool_1")

          self.conv2 =  self.conv_layer(self.conv1, 16, 32, "conv2",filter_size=5,padding="SAME")
          self.conv2 =  batch_norm(name='bn_conv2')(self.conv2, phase)
          self.conv2 = self.max_pool(self.conv2,"max_pool_2")
          
          self.conv3 = self.conv_layer(self.conv2, 32, 32, "conv3", filter_size=3, padding="SAME")
          self.conv3 = batch_norm(name="bn_conv3")(self.conv3, phase)
          self.conv3 = self.max_pool(self.conv3, "max_pool_3")

          self.conv4 = self.conv_layer(self.conv3, 32, 64, "conv4", filter_size=3, padding="SAME")
          self.conv4 = batch_norm(name="bn_conv4")(self.conv4, phase)
          self.conv4 = self.max_pool(self.conv4, "max_pool_4")

          print "After Maxpool4: " + str(self.conv4.get_shape())

          if phase:
               self.conv4_dropout = self.dropout(self.conv4, 0.5, "dropout")
          else:
               self.conv4_dropout = self.dropout(self.conv4, 1, "dropout")

          print "After dropout: " + str(self.conv4_dropout.get_shape())

          self.fc_1 = self.fc_layer(self.conv4_dropout,32*32*64,3,"fc")

          self.output = self.fc_1

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

          # will return total batch error -> needed in validation case
          xerror_batch = tf.square(x_error_abs)
          yerror_batch = tf.square(y_error_abs)
          theta_error1 = tf.square(theta_error_abs) 
                         
          x_mse = tf.reduce_mean(xerror_batch)
          y_mse = tf.reduce_mean(yerror_batch)
          theta_mse = tf.reduce_mean(theta_error1) 
          weighted_mse = weights[0]*x_mse + weights[1]*y_mse + weights[2]*theta_mse
                         
          x_mae = tf.reduce_mean(x_error_abs)
          y_mae = tf.reduce_mean(y_error_abs)
          theta_mae = tf.reduce_mean(theta_error_abs)
          weighted_mae = weights[0]*x_mae + weights[1]*y_mae + weights[2]*theta_mae

          with tf.variable_scope("output_loss"):

               tf.summary.scalar("x_mse", x_mse)
               tf.summary.scalar("y_mse", y_mse)
               tf.summary.scalar("theta_mse", theta_mse)
               tf.summary.scalar("loss_mse", weighted_mse)
               
               tf.summary.scalar("x_mae", x_mae)
               tf.summary.scalar("y_mae", y_mae)
               tf.summary.scalar("theta_mae", theta_mae)
               tf.summary.scalar("loss_mae", weighted_mae)
               
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
     
          return weighted_mse

     def batch_loss(self, output, label):

          self.label = label
          # squared_error = tf.losses.mean_squared_error(output , label)
          weights = self.weights

          # Calculate both abs and sqr for MAE and RMS respec
          x_error_abs = tf.abs(output[:,0] - label[:,0])
          y_error_abs = tf.abs(output[:,1] - label[:,1])
          theta_error_abs = tf.minimum( tf.abs(output[:,2] - label[:,2]), tf.abs(output[:,2] + 3.6 - label[:,2]) )
          theta_error_abs = tf.minimum( theta_error_abs, tf.abs(output[:,2] - 3.6 - label[:,2]) ) 
          
          # will return total batch error -> needed in validation case
          x_batch_error_sqr = tf.reduce_sum( tf.square(x_error_abs) )
          y_batch_error_sqr = tf.reduce_sum( tf.square(y_error_abs) )
          theta_batch_error_sqr = tf.reduce_sum( tf.square(theta_error_abs) )
          weighted_error_sqr = weights[0]*x_batch_error_sqr + weights[1]*y_batch_error_sqr + weights[2]*theta_batch_error_sqr

          x_batch_error_abs = tf.reduce_sum(x_error_abs)
          y_batch_error_abs = tf.reduce_sum(y_error_abs)
          theta_batch_error_abs = tf.reduce_sum(theta_error_abs)
          weighted_error_abs = weights[0]*x_batch_error_abs + weights[1]*y_batch_error_abs + weights[2]*theta_batch_error_abs
          
          return (weighted_error_sqr, weighted_error_abs, x_batch_error_sqr, x_batch_error_abs, y_batch_error_sqr, y_batch_error_abs, theta_batch_error_sqr, theta_batch_error_abs)

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

