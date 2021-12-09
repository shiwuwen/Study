import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops 

ops.reset_default_graph()

sess = tf.Session()

# knn test
k = 4
batch_size = 8
train_nums = 1000
test_nums = 100

# mnist.train.images: [55000, 784]
# mnist.test.images: [10000, 784]
mnist = input_data.read_data_sets('mnist/', one_hot=True)
train_data = mnist.train.images
test_data = mnist.test.images
train_labels = mnist.train.labels
test_labels = mnist.test.labels

random_train_indices = np.random.choice(len(train_data), train_nums, replace=False)
random_test_indices = np.random.choice(len(test_data), test_nums, replace=False)

x_vals_train = train_data[random_train_indices]
x_vals_test = test_data[random_test_indices]
y_vals_train = train_labels[random_train_indices]
y_vals_test = test_labels[random_test_indices]

## tensorflow code
x_data_train = tf.placeholder(shape=[None, 784], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None, 10], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None, 10], dtype=tf.float32)

hyper_parameter_k = tf.placeholder(shape=(), dtype=tf.int32)

# L2 distance
distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), axis=2))

top_k_vals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=hyper_parameter_k)
prediction_1 = tf.gather(y_target_train, top_k_indices)
count_of_prediction = tf.reduce_sum(prediction_1, axis=1)
prediction = tf.argmax(count_of_prediction, axis=1)

accuracy = tf.truediv(tf.reduce_sum(tf.cast(tf.equal(prediction, tf.argmax(y_target_test, axis=1)), dtype=tf.int32)), tf.shape(y_target_test)[0])

for k in range(1, 6):
    num_loops = int(np.ceil(test_nums / batch_size))
    total_prediction = []
    total_truth = []
    for step in range(num_loops):
        min_index = step * batch_size
        max_index = min((step+1) * batch_size, test_nums)
        x_batch = x_vals_test[min_index : max_index]
        y_batch = y_vals_test[min_index : max_index]

        predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch, hyper_parameter_k: k})

        curr_accuracy = sess.run(accuracy, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch, y_target_train: y_vals_train, y_target_test: y_batch, hyper_parameter_k: k})

        total_prediction.extend(predictions)
        total_truth.extend(np.argmax(y_batch, axis=1))

        # print('loop: ', step, ", current prediction is: ", predictions)
        # print('loop: ', step, ",      current truth is: ", np.argmax(y_batch, axis=1))
        # print('loop: ', step, ",   current accuracy is: ", curr_accuracy)

    total_accuracy = sum([1./test_nums for index in range(test_nums) if total_prediction[index] == total_truth[index]])

    print('currrnt k is : ', k , ' total accuracy is : ', total_accuracy)