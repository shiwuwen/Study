import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data 

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001

mnist = input_data.read_data_sets('./mnist', one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

print(mnist.train.images.shape)
print(mnist.train.labels.shape)

tf_x = tf.placeholder(tf.float32, [None, 28*28], name='input_x') /255.
image = tf.reshape(tf_x, [-1, 28, 28, 1], name='input_reshape')
tf_y = tf.placeholder(tf.int32, [None, 10], name='input_y')

conv1 = tf.layers.conv2d(
	inputs=image,
	filters=16,
	kernel_size=5,
	strides=1,
	padding='same',
	activation=tf.nn.relu
)

pool1 = tf.layers.max_pooling2d(
	conv1,
	pool_size=2,
	strides=2
)

conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
flat = tf.reshape(pool2, [-1, 7*7*32])
output = tf.layers.dense(flat, 10, name='pred_y')

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y,axis=1), predictions=tf.argmax(output, axis=1),)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

saver = tf.train.Saver(max_to_keep=1)


for step in range(600):
	b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
	_, loss_ = sess.run([train_op, loss], feed_dict={tf_x:b_x, tf_y:b_y})

	if step % 50 == 0:
		saver.save(sess, "./checkpoint/mycheckpoint", global_step=step)
		accuracy_ = sess.run(accuracy, feed_dict={tf_x:b_x, tf_y:b_y})
		print('Step : {}, train_loss: {:.2f}, accuracy: {}'.format(step, loss_, accuracy_))

