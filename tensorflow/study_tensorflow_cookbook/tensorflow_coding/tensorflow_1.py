import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops.gen_array_ops import shape, size
import requests

sess = tf.Session()

# test 4
# Load the data
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
num_features = len(cols_used)
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]

y_vals = np.transpose([np.array([y[13] for y in housing_data])])
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

## Min-Max Scaling
# x_vals.ptp(0): 返回 max - min
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0) 

y_vals = np.transpose([np.array([y[13] for y in housing_data])])
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

## Min-Max Scaling
# x_vals.ptp(0): 返回 max - min
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

# Split the data into train and test sets
np.random.seed(13)  #make results reproducible
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare k-value and batch size
k = 4
batch_size = 20 #len(x_vals_test)

# Placeholders
x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Declare distance metric
# L1
distance = tf.reduce_sum(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1)), axis=2)
# temp = sess.run(distance)
# print(sess.run(tf.shape(temp)))

top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
top_k_xvals = tf.truediv(1.0, top_k_xvals)
x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, axis=1), axis=1)
x_sums_repeated = tf.matmul(x_sums, tf.ones([1, k], tf.float32))
x_sums_weight = tf.expand_dims(tf.div(top_k_xvals, x_sums_repeated), axis=1)

top_k_yvals = tf.gather(y_target_train, top_k_indices)

prediction = tf.squeeze(tf.matmul(x_sums_weight, top_k_yvals), axis=1)

mse = tf.div(tf.reduce_sum(tf.subtract(prediction, y_target_test)), batch_size)

num_loops = int(np.ceil(len(x_vals_test) / batch_size))

for step in range(num_loops):
    min_index = step * batch_size
    max_index = min((step + 1) * batch_size, len(x_vals_test))
    x_batch = x_vals_test[min_index : max_index]
    y_batch = y_vals_test[min_index : max_index]

    predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                         y_target_train: y_vals_train, y_target_test: y_batch})
    batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch})

    print('Batch #' + str(step+1) + ' MSE: ' + str(np.round(batch_mse,3)))


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