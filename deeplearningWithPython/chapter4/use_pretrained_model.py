from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers


#####不使用数据增强的快速特征提取##########
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
def feature_extraction_without_dataaugmentation(conv_base):
	base_dir = '/home/shiwuwen/workplace/dataset/dogs_and_cats/pritical_dataset'
	train_dir = os.path.join(base_dir, 'train')
	validation_dir = os.path.join(base_dir, 'validation')
	test_dir = os.path.join(base_dir, 'test')

	datagen = ImageDataGenerator(rescale=1./255)
	batch_size = 20

	train_features, train_labels = extract_features(conv_base, train_dir, datagen, 2000)
	validation_features, validation_labels = extract_features(conv_base, train_dir, datagen, 1000)
	test_features, test_labels = extract_features(conv_base, train_dir, datagen, 1000)

	train_features = np.reshape(train_features, (2000,4*4*512))
	validation_features = np.reshape(validation_features, (1000,4*4*512))
	test_features = np.reshape(test_features, (1000,4*4*512))

	model = models.Sequential()
	model.add(layers.Dense(256, activation='relu', input_shape=(4*4*512,)))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation='sigmoid'))

	model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
					loss='binary_crossentropy',
					metrics=['accuracy'])

	history = model.fit(train_features, train_labels,
						epochs=30,
						batch_size=20,
						validation_data=(validation_features, validation_labels))

	loss, acc = model.evaluate(test_features, test_labels)

	print('loss: ', loss)
	print('acc: ', acc)

	return model

def extract_features(conv_base, directory, datagen, sample_count=2000, batch_size=20):
	features = np.zeros(shape=(sample_count, 4, 4, 512))
	labels = np.zeros(shape=(sample_count))

	generator = datagen.flow_from_directory(
		directory,
		target_size=(150,150),
		batch_size=batch_size,
		class_mode='binary')
	i = 0
	for inputs_batch, label_batch in generator:
		features_batch = conv_base.predict(inputs_batch)
		# print(features_batch.shape)
		# print(label_batch.shape)
		features[i*batch_size:(i+1)*batch_size] = features_batch
		labels[i*batch_size:(i+1)*batch_size] = label_batch
		i += 1
		if i*batch_size>=sample_count:
			break

	return features, labels

######end#########


####使用数据增强的特征提取######
def feature_extract_with_dataaugmentation(conv_base):

	conv_base.trainable = False

	model = models.Sequential()
	model.add(conv_base)
	model.add(layers.Flatten())
	model.add(layers.Dense(256, activation='relu'))
	# model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation='sigmoid'))


	base_dir = '/home/shiwuwen/workplace/dataset/dogs_and_cats/pritical_dataset'
	train_dir = os.path.join(base_dir, 'train')
	validation_dir = os.path.join(base_dir, 'validation')
	test_dir = os.path.join(base_dir, 'test')

	train_datagen = ImageDataGenerator(
 					rescale=1./255,
 					rotation_range=40,
 					width_shift_range=0.2,
 					height_shift_range=0.2,
 					shear_range=0.2,
 					zoom_range=0.2,
 					horizontal_flip=True,
 					fill_mode='nearest')

	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
		train_dir, target_size=(150,150), 
		batch_size=32, class_mode='binary')

	validation_generator= test_datagen.flow_from_directory(
		validation_dir, target_size=(150,150),
		batch_size=32, class_mode='binary')

	test_genarator = test_datagen.flow_from_directory(
		test_dir, target_size=(150,150),
		batch_size=32, class_mode='binary')

	model.compile(loss='binary_crossentropy',
		optimizer=optimizers.RMSprop(lr=2e-5),
		metrics=['acc'])

	history = model.fit_generator(
		train_generator,
		steps_per_epoch=100,
		epochs=5,
		validation_data=validation_generator,
		validation_steps=50)

	loss, acc = model.evaluate_generator(test_genarator)

	print('loss: ', loss)
	print('acc: ', acc)

	return history
########end#########

if __name__ == '__main__':
	conv_base = VGG16(weights='imagenet',
					include_top=False,
					input_shape=(150,150,3))
	# print(conv_base.summary())
	# print(conv_base.layers[0].name)
	# feature_extraction_without_dataaugmentation(conv_base)
	feature_extract_with_dataaugmentation(conv_base)

