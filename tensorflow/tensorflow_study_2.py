import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

#创建一个全连接层，可用 tf.layers.dense(inputs, out_size, activation_function)代替
def add_layer(inputs, in_size, out_size, activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]))
	Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)

	return outputs

x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
# print("y_data： ", y_data.shape)

#定义placeholder，值待传入
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

#输出层
prediction = add_layer(l1, 10, 1)

#计算均方差损失，可用loss = tf.losses.mean_squared_error(ys, prediction)代替
loss1 = tf.reduce_mean(tf.square(ys-prediction))
# loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     # reduction_indices=[1]))


optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss1)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x_data, y_data)
	plt.ion()
	plt.show()

	for step in range(1000):
		sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

		if step % 50 ==0:

			print("loss1: ", sess.run(loss1, feed_dict={xs: x_data, ys: y_data}))
			# print("loss2: ", sess.run(loss2, feed_dict={xs: x_data, ys: y_data}))
			try:
				ax.lines.remove(lines[0])
			except Exception:
				pass
			prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
			lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
			plt.pause(1)
