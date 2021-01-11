from keras import models
from keras.preprocessing.image import ImageDataGenerator

import os

model = models.load_model("dogs_and_cats_small_1.h5")
print(model.summary())

base_dir = '/home/shiwuwen/workplace/dataset/dogs_and_cats/pritical_dataset'
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
					test_dir, target_size=(150,150), 
					batch_size=20, class_mode='binary')

loss, acc = model.evaluate_generator(test_generator)

print('loss: ', loss)
print('acc: ', acc)


'''
str has no attribute decode
/home/shiwuwen/anaconda3/envs/wsw/lib/python3.6/site-packages/keras/models.py
line 273 293 322
/home/shiwuwen/anaconda3/envs/wsw/lib/python3.6/site-packages/keras/engine/topology.py 
line 2940 2945 3339 3345
'''