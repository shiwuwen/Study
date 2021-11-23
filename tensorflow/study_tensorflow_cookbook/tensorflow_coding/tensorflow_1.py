import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_array_ops import size

sess = tf.Session()

# # test2
# x_shape = [1, 4, 4, 1]
# x_val = np.random.uniform(size=x_shape)
# x_data = tf.placeholder(dtype=tf.float32, shape=x_shape)

# my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
# my_strides = [1, 2, 2, 1]

# # tf.nn.conv2d(input, filter, strides, padding, data_format, dilations, name)  
# my_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='layer1')

# def custom_layer(input_layer):
#     # delete where dim == 1
#     squeeze_input_layer = tf.squeeze(input_layer)
#     A = tf.constant([[1., 2.], [-1., 3.]])
#     b = tf.constant(1., shape=[2, 2])

#     y_ = tf.matmul(A, squeeze_input_layer)
#     y = tf.add(y_, b)

#     return tf.sigmoid(y)

# with tf.name_scope('custom_layer') as scope:
#     custom_layer1 = custom_layer(my_avg_layer)

# print(sess.run(my_avg_layer, feed_dict={x_data : x_val}))

# print(sess.run(custom_layer1, feed_dict={x_data : x_val}))


# # test1
# x_vals = [1., 3., 5., 7., 9.]
# # tf.placeholder(dtype, shape, name)
# x_data = tf.placeholder(tf.float32)
# # tf.constant(value, shape, name)
# m_const = tf.constant(3.)

# y = tf.multiply(x_data, m_const)

# for x_val in x_vals:
#     print(sess.run(y, feed_dict={x_data : x_val}))