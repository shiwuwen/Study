import keras
from keras import models 
from keras import Input, layers
import numpy as np 

def squential_model_with_keras_api():

	input_tensor = Input(shape=(64,))
	dense_1 = layers.Dense(32, activation='relu')(input_tensor)
	dense_2 = layers.Dense(32, activation='relu')(dense_1)
	output_tensor = layers.Dense(10, activation='softmax')(dense_2)

	model = models.Model(input_tensor, output_tensor)

	model.compile(optimizer='rmsprop',
					loss='categorical_crossentropy',
					metrics=['accuracy'])

	print(model.summary())


def multiple_input_network():
	text_vocabulary_size = 10000
	question_vocabulary_size = 10000
	answer_vocabulary_size = 500

	text_input = Input(shape=(None,), dtype='int32', name='text')
	embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
	encoded_text = layers.LSTM(32)(embedded_text)

	question_input = Input(shape=(None,), dtype='int32', name='question')
	embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
	encoded_question = layers.LSTM(16)(embedded_question)

	concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

	answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

	model = models.Model([text_input, question_input], answer)

	model.compile(optimizer='rmsprop',
					loss='categorical_crossentropy',
					metrics=['acc'])

	num_samples = 1000
	max_length = 100

	text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
	question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
	answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
	answers = keras.utils.to_categorical(answers, answer_vocabulary_size)
	
	# model.fit([text, question], answers, epochs=10, batch_size=128)
	model.fit({'text': text, 'question': question}, answers, epochs=20, batch_size=128)


	result = model.predict([text[:1], question[:1]])

	print(np.argmax(result[0]))
	print(np.argmax(answers[0]))

if __name__ == '__main__':
	# squential_model_with_keras_api()

	multiple_input_network()
 
