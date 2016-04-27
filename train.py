import numpy
import scipy
import gensim
import skimage
import theano
import nltk
import os

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Embedding, Siamese

# Dimension placeholders
V = 86570  # vocabulary size
L = 2470 # total number of words from every review
D = 30

# review_matrix_one_hot = review_to_onehot_matrix(review, my_one_hot)
# weights = np.random.uniform(-1, 1, (V,D,L))
# embedded_matrix = one_hot_input * weights



model.add(Embedding(input_dim=V, output_dim=L, init='uniform', weights=weights))

# model.add(Dense(64, input_dim=L, init='uniform'))




model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd)

model.fit(X_train, y_train,
          nb_epoch=20,
          batch_size=16,
          show_accuracy=True)
score = model.evaluate(X_test, y_test, batch_size=16)


