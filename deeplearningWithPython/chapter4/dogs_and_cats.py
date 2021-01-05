import os, shutil

def make_dataset():
	'''
	从原始数据集中提取部分数据
	'''
	original_dataset_dir = '/home/shiwuwen/workplace/dataset/dogs_and_cats/train/'

	base_dir = '/home/shiwuwen/workplace/dataset/dogs_and_cats/pritical_dataset'
	os.mkdir(base_dir)

	train_dir = os.path.join(base_dir, 'train')
	os.mkdir(train_dir)
	validation_dir = os.path.join(base_dir, 'validation')
	os.mkdir(validation_dir)
	test_dir = os.path.join(base_dir, 'test')
	os.mkdir(test_dir)

	train_cats_dir = os.path.join(train_dir, 'cats')
	os.mkdir(train_cats_dir)
	train_dogs_dir = os.path.join(train_dir, 'dogs')
	os.mkdir(train_dogs_dir)

	validation_cats_dir = os.path.join(validation_dir, 'cats')
	os.mkdir(validation_cats_dir)
	validation_dogs_dir = os.path.join(validation_dir, 'dogs')
	os.mkdir(validation_dogs_dir)

	test_cats_dir = os.path.join(test_dir, 'cats')
	os.mkdir(test_cats_dir)
	test_dogs_dir = os.path.join(test_dir, 'dogs')
	os.mkdir(test_dogs_dir)

	fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(train_cats_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(validation_cats_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(test_cats_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(train_dogs_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(validation_dogs_dir, fname)
		shutil.copyfile(src, dst)

	fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, fname)
		dst = os.path.join(test_dogs_dir, fname)
		shutil.copyfile(src, dst)

from keras import models
from keras import layers
from keras import optimizers
def build_network(return_model_struct=False):
	'''
	构建卷积神经网络
	'''
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(64, (3,3), activation='relu'))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(128, (3,3), activation='relu'))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(128, (3,3), activation='relu'))
	model.add(layers.MaxPooling2D((2,2)))

	model.add(layers.Flatten())
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
					optimizer=optimizers.RMSprop(lr=1e-4),
					metrics=['accuracy'])

	if return_model_struct == True:
		return model, model.summary()
	else:
		return model

from keras.preprocessing.image import ImageDataGenerator
def data_process(train_dir, validation_dir):
	'''
	使用python生成器生成训练和验证数据集
	'''
	train_datagen = ImageDataGenerator(rescale=1./255)
	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
		train_dir, target_size=(150,150), 
		batch_size=20, class_mode='binary')

	validation_generator= test_datagen.flow_from_directory(
		validation_dir, target_size=(150,150),
		batch_size=20, class_mode='binary')

	return train_generator, validation_generator

def train_model(model, train_generator, validation_generator):
	'''
	训练模型并保存权重数据
	'''
	history = model.fit_generator(
		train_generator,
		steps_per_epoch=100,
		epochs=30,
		validation_data=validation_generator,
		validation_steps=50)
	model.save('dogs_and_cats_small_1.h5')

	return history

import matplotlib.pyplot as plt
def draw_loss_accuracy(history):
	'''
	绘制损失和正确率曲线
	'''
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	# make_dataset()
	
	model, model_struct = build_network(True)
	# print(model_struct)
	base_dir = '/home/shiwuwen/workplace/dataset/dogs_and_cats/pritical_dataset'
	train_dir = os.path.join(base_dir, 'train')
	validation_dir = os.path.join(base_dir, 'validation')

	train_generator, validation_generator = data_process(train_dir, validation_dir)
	# for data_batch, label_batch in train_generator:
	# 	print(data_batch.shape)
	# 	print(label_batch[0])
	# 	break

	history = train_model(model, train_generator, validation_generator)
	draw_loss_accuracy(history)
