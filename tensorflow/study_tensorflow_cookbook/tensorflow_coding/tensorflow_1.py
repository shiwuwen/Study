import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops.gen_array_ops import size

sess = tf.Session()


# # test 3
# print('_______ regression __________')
# # np.random.normal(loc, scale, shape)
# x_vals = np.random.normal(1, 0.1, 100)
# # np.repeat(value, repeats)
# y_vals = np.repeat(10., 100)
# x_data = tf.placeholder(dtype=tf.float32, shape=[1])
# y_target = tf.placeholder(dtype=tf.float32, shape=[1])

# # tf.Variable(initializer, name)
# # tf.random_normal(shape, mean, stddev, dtype, seed, name)
# A = tf.Variable(tf.random_normal(shape=[1]))

# pre_out = tf.multiply(x_data, A)

# loss = tf.square(pre_out - y_target)

# opt = tf.train.GradientDescentOptimizer(0.02)
# train_opt = opt.minimize(loss)

# sess.run(tf.global_variables_initializer())

# for step in range(2001):
#     random_index = np.random.choice(len(x_vals))
#     random_x = [x_vals[random_index]]
#     random_target = [y_vals[random_index]]

#     sess.run(train_opt, feed_dict={x_data : random_x, y_target : random_target})

#     if step % 200 == 0:
#         print("value A is : " + str(sess.run(A)))
#         print("loss is : " + str(sess.run(loss, feed_dict={x_data : random_x, y_target : random_target})))

# print("_______ classification __________")
# ops.reset_default_graph()

# sess = tf.Session()

# x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
# y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))

# x_data = tf.placeholder(dtype=tf.float32, shape=[1])
# y_target = tf.placeholder(dtype=tf.float32, shape=[1])

# A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# pre_out = tf.add(x_data, A)

# pre_out_expand_dim = tf.expand_dims(pre_out, 0)
# y_target_expand_dim = tf.expand_dims(y_target, 0)

# sess.run(tf.global_variables_initializer())

# loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pre_out_expand_dim, labels=y_target_expand_dim)

# opt = tf.train.GradientDescentOptimizer(0.05)
# train_op = opt.minimize(loss)



# for step in range(1000):
#     rand_index = np.random.choice(100)
#     rand_x = [x_vals[rand_index]]
#     rand_y = [y_vals[rand_index]]
    
#     sess.run(train_op, feed_dict={x_data: rand_x, y_target: rand_y})
#     if (step + 1) % 200 == 0:
#         print('Step #' + str(step+1) + ' A = ' + str(sess.run(A)))
#         print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))


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