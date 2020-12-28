from keras.datasets import reuters
from keras import models
from keras import layers

import numpy as np 
import matplotlib.pyplot as plt 

'''
路透社新闻分类数据集
训练样本数：8982
测试样本数：2246
分类数： 46
'''

def get_data_set():

	(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

	return train_data, train_labels, test_data, test_labels

def num2word(input_data):
	word_index = reuters.get_word_index()

	print('字典长度： ' + str(len(word_index)))

	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

	result = ''.join(reverse_word_index.get(i-3, '?') for i in input_data)

	return result

def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension), dtype=np.float32)

	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.

	return results

def bulid_network():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(46, activation='softmax'))

	model.compile(optimizer='rmsprop',
					loss='categorical_crossentropy',
					metrics=['accuracy'])

	return model

def train_model(model, x_data, x_labels, epochs=20, batch_size=512):
	val_x_data = x_data[:1000]
	partical_x_data = x_data[1000:]

	val_x_labels = x_labels[:1000]
	partical_x_labels = x_labels[1000:]

	history = model.fit(partical_x_data, partical_x_labels, epochs=epochs, batch_size=batch_size,
				validation_data=[val_x_data, val_x_labels])

	return history

def draw_training_and_validation_loss(loss, val_loss):
	epochs = range(1, len(loss)+1)

	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')

	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss') 
	plt.legend()

	plt.show()

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



if __name__ == '__main__':

	train_data, train_labels, test_data, test_labels = get_data_set()

	# print('训练集长度： ' + str(len(train_data)))
	# print('测试及长度： ' + str(len(test_data)))
	# print(train_data[0])

	# print(num2word(train_data[0]))

	x_data = vectorize_sequences(train_data)
	x_labels = vectorize_sequences(train_labels, 46)

	y_data = vectorize_sequences(test_data)
	y_labels = vectorize_sequences(test_labels, 46)

	# print(x_labels[:10])
	# print(train_labels[:10])

	model = bulid_network()
	history = train_model(model, x_data, x_labels, epochs=20)

	history_dic = history.history
	# print(history_dic.keys())

	# loss = history_dic['loss']
	# val_loss = history_dic['val_loss']
	# acc = history_dic['acc']
	# val_acc = history_dic['val_acc']
	# draw_training_and_validation_loss(loss, val_loss)
	# draw_training_and_validation_accuracy(acc, val_acc)

	val_loss, val_accuracy = model.evaluate(y_data, y_labels)

	print(val_accuracy)

	predict_result = model.predict(y_data)

	print(np.argmax(predict_result[0]))