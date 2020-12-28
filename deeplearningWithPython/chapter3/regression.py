from keras.datasets import boston_housing
from keras import models, layers

import numpy as np 

'''
波士顿放假数据集
训练样本数： 404 
测试样本数： 102
特征数： 13
'''

def get_data_set():

	(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

	return train_data, train_targets, test_data, test_targets

def data_standardization(input_data):
	'''
	数据标准化
	'''
	mean = input_data.mean(axis=0)
	input_data -= mean
	std = input_data.std(axis=0)
	input_data /= std

	return input_data, mean, std

def build_network():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu', input_shape=(13,)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))

	model.compile(optimizer='rmsprop',
					loss='mse',
					metrics=['mae'])

	return model

def train_model_with_K_fold_validation(train_data, train_targets, model, k=4, epochs=100, batch_size=1):
	'''
	使用K折交叉验证训练网络
	'''
	num_val_samples = len(train_data) // k
	num_epochs = epochs
	all_mae_history = []

	for i in range(k):
		print('processing fold #', i)
		val_data = train_data[i*num_val_samples: (i+1)*num_val_samples]
		val_targets = train_targets[i*num_val_samples: (i+1)*num_val_samples]

		#numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接。其中a1,a2,...是数组类型的参数
		partical_train_data = np.concatenate((train_data[: i*num_val_samples], 
												train_data[(i+1)*num_val_samples : ]), axis=0)
		partical_train_targets = np.concatenate([train_targets[ : i*num_val_samples],
												train_targets[(i+1)*num_val_samples : ]], axis=0)

		history = model.fit(partical_train_data, partical_train_targets, epochs=num_epochs, 
			batch_size=batch_size, validation_data=(val_data,val_targets), verbose=0)

		mae_history = history.history['val_mean_absolute_error']

		all_mae_history.append(mae_history)

	return all_mae_history


if __name__ == '__main__':

	train_data, train_targets, test_data, test_targets = get_data_set()

	# print(len(train_data))
	# print(len(test_data))
	# print(train_data.shape)
	# print(train_targets[0])
	# print(test_data[0])

	#数据标准化
	train_data, mean, std = data_standardization(train_data)
	test_data -= mean
	test_data /= std

	model = build_network()
	all_mae_history = train_model_with_K_fold_validation(train_data, train_targets, model)

	# print(len(all_mae_history))
	# print(all_mae_history[0])

	predict_result = model.predict(np.expand_dims(test_data[0], axis=0))

	predict = model.predict(test_data[0:1])

	print(predict_result)

