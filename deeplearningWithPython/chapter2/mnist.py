from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

def get_data_set():
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

	return train_images, train_labels, test_images, test_labels

def build_network():
	model = models.Sequential()
	model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
	model.add(layers.Dense(10, activation='softmax'))

	model.compile(optimizer='rmsprop',
					loss='categorical_crossentropy',
					metrics=['accuracy'])

	return model

def train_model(model, train_images, train_labels, epochs=5, batch_size=128):
	val_images = train_images[:5000]
	partical_train_images = train_images[5000:]

	val_labels = train_labels[:5000]
	partical_train_labels = train_labels[5000:]

	history = model.fit(partical_train_images, partical_train_labels, epochs=epochs, batch_size=batch_size,
				validation_data=(val_images, val_labels))

	return history


if __name__ == '__main__':

	train_images, train_labels, test_images, test_labels = get_data_set()

	print('train_images 形状： ' + str(train_images.shape))
	print('train_images 长度： ' + str(len(train_images)))
	# print('train_images 大小： ' + str(len(train_images[0])))
	print('test_images 长度: ' + str(len(test_images)))

	train_images = train_images.reshape((60000, 28*28))
	train_images = train_images.astype('float32') / 255

	test_images = test_images.reshape((10000, 28*28))
	test_images = test_images.astype('float32') / 255

	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)

	model = build_network()

	history = train_model(model, train_images, train_labels)
	history_dic = history.history

	test_result = model.evaluate(test_images, test_labels)

	print('测试结果： ', end='')
	print(test_result)