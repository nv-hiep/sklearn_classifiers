import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense


imdb_dir = 'data/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts  = []


for label_type in ['neg', 'pos']:
	dir_name = os.path.join(train_dir, label_type) # 2 dirs: neg/ and pos/
	val = 1 if (label_type == 'pos') else 0
	for fname in os.listdir(dir_name):
		if fname[-4:] != '.txt':
			continue

		f = open(os.path.join(dir_name, fname))
		texts.append(f.read())
		f.close()
		labels.append( val )
		# End - if

	# End - for
# End - for


sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)


model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)