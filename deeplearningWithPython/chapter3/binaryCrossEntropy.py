from keras.datasets import imdb
import numpy as np 
from keras import models
from keras import layers

import matplotlib.pyplot as plt 

import time

'''
互联网电影数据库（IMDB） 电影评论分类
训练样本数：25000
测试样本数：25000
类别数：2 正面评论，负面评论
'''

def get_data_set():
	'''
	获取IMDB数据集，它包含来自互联网电影数据库（IMDB）的 50 000 条严重两极分
	化的评论。数据集被分为用于训练的 25 000 条评论与用于测试的 25 000 条评论，
	训练集和测试集都包含 50% 的正面评论和 50% 的负面评论。
	'''
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

	return train_data, train_labels, test_data, test_labels

def num2word(input_data):
	'''
	将整数表示的句子向量转换为具体单词
	'''
	#获取单词与整数的映射 （单词，整数）
	word_index = imdb.get_word_index()
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

	#获取整数对应的单词
	decoded_review= ''.join([reverse_word_index.get(i-3, '?') for i in input_data])

	#返回转换后的结果
	return decoded_review

def vectorize_sequences(sequences, dimension=10000):
	'''
	将原始整数向量转换为one-hot向量
	'''
	results = np.zeros((len(sequences), dimension), dtype=np.float32)

	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.

	return results

def build_network():
	'''
	搭建网络
	'''
	model = models.Sequential()
	model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
	model.add(layers.Dense(16, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.compile(optimizer='rmsprop',
					loss='binary_crossentropy',
					metrics=['accuracy'])

	return model

def train_model(model, x_train, y_train, epochs=20, batch_size=512):
	'''
	训练网络
	'''
	#将数据分为测试集和验证集
	x_val = x_train[:1000]
	partial_x_train = x_train[1000:]

	y_val = y_train[:1000]
	partial_y_train = y_train[1000:]

	#history保留训练过程的信息 [val_loss, val_acc, loss, acc]
	history = model.fit(partial_x_train, 
						partial_y_train, 
						epochs=epochs,
						batch_size=batch_size,
						validation_data=(x_val, y_val))

	return history


def draw_training_and_validation_loss(loss, val_loss):
	'''
	绘制训练和验证损失图
	用于查看训练是否过拟合
	'''
	epochs = range(1, len(loss)+1)

	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and Validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()

	plt.show()

	# time.sleep(10)

	# plt.clf()


def draw_training_and_validation_accuracy(acc, val_acc):
	'''
	绘制训练和验证准确率图
	'''
	epochs = range(1, len(acc)+1)

	plt.plot(epochs, acc, 'bo', label='Training accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
	plt.title('Training and Validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('accuracy')
	plt.legend()

	plt.show()

	# time.sleep(10)

	# plt.clf()


if __name__ == '__main__':
	#获取数据集
	train_data, train_labels, test_data, test_labels = get_data_set()

	#将数据转换为one-hot向量
	x_train = vectorize_sequences(train_data)
	x_test = vectorize_sequences(test_data)

	y_train = np.asarray(train_labels).astype('float32')
	y_test = np.asarray(test_labels).astype('float32')

	#获得模型
	current_model = build_network()

	#训练模型
	history = train_model(current_model, x_train, y_train)

	#获取历史信息
	history_dic = history.history

	loss = history_dic['loss']
	val_loss = history_dic['val_loss']
	acc = history_dic['acc']
	val_acc = history_dic['val_acc']

	#绘图
	draw_training_and_validation_loss(loss, val_loss)
	draw_training_and_validation_accuracy(acc, val_acc)

	#在测试集上验证结果
	predict_result = current_model.evaluate(x_test, y_test)

	print(predict_result)

	# str_result = num2word(train_data[0])
	# print(str_result)
	# print(len(train_data[1]))