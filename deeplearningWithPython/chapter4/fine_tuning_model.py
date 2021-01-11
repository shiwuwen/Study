from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def build_model(conv_base):
	conv_base.trainable = True 
	set_trainable = False

	for layer in conv_base.layers:
		if layer.name == 'block5_conv1':
			set_trainable = True

		if set_trainable:
			layer.trainable = False
		else:
			layer.trainable = True

	model = models.Sequential()
	model.add(conv_base)
	model.add(layers.Flatten())
	model.add(layers.Dense(256, activation='relu'))
	# model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
					optimizer=optimizers.RMSprop(lr=1e-5),
					metrics=['accuracy'])

	return model

def train_model(model):

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

	model.fit_generator(train_generator,
						steps_per_epoch=100,
						epochs=100,
						validation_data=validation_generator,
						validation_steps=50)

	loss, acc = model.evaluate_generator(test_genarator)

	print('loss: ', loss)
	print('acc: ', acc)




if __name__ == '__main__':
	conv_base = VGG16(weights='imagenet',
						include_top=False,
						input_shape=(150,150,3))
	model = build_model(conv_base)

	print(model.summary())

	train_model(model)

	model.save('fine_tuning_model.h5')