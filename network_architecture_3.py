from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers.core import (Dense, Dropout,
                               Activation, Flatten)
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
import theano
import cPickle
from keras.regularizers import l2, l1, l1l2
from convert_review import build_design_matrix
import modelParameters
import os
import datetime

DEVSPLIT = 14
USEWORDS = True

if USEWORDS:
	VocabSize = modelParameters.VocabSize_w
	maxReviewLen = modelParameters.MaxLen_w
	skipTop = modelParameters.skip_top
else:
	VocabSize = modelParameters.VocabSize_c
	maxReviewLen = modelParameters.MaxLen_c
	skipTop = 0

batch_size = 80

num_filters1 = 380
filter_length1 = 2
stride_len1 = 1
pool_len1 = 2

num_filters2 = 160
filter_length2 = 3
stride_len2 = 1
pool_len2 = 2

embedding_dims = 220

hidden_dims = 100
num_epochs = 3

print( 'Loading data...' )

((X_train, y_train), (X_test, y_test)) = build_design_matrix( VocabSize,
                                                              use_words = USEWORDS,
                                                              skip_top = skipTop,
                                                              dev_split = DEVSPLIT )

print( len( X_train ), 'train sequences' )
print( len( X_test ), 'test sequences' )

print( 'X_train shape:', X_train.shape )
print( 'X_test shape:', X_test.shape )

print( 'Build model...' )
model = Sequential( )

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add( Embedding( VocabSize, embedding_dims, input_length = maxReviewLen ) )

model.add( Dropout( 0.10 ) )
# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add( Convolution1D( nb_filter = num_filters1,
                          filter_length = filter_length1,
                          border_mode = 'valid',
                          activation = 'relu',
                          subsample_length = stride_len1,
                          ) )

# input_length=maxReviewLen, input_dim=VocabSize

model.add( Dropout( 0.25 ) )
model.add( MaxPooling1D( pool_length = 2 ) )

model.add( Convolution1D( nb_filter = num_filters2,
                          filter_length = filter_length2,
                          border_mode = 'valid',
                          activation = 'relu',
                          subsample_length = stride_len2,

                          ) )

model.add( Dropout( 0.40 ) )

# we use standard max pooling (halving the output of the previous layer):
model.add( MaxPooling1D( pool_length = 2 ) )

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add( Flatten( ) )

# We add a vanilla hidden layer:
model.add( Dense( hidden_dims, W_regularizer = None ) )

model.add( Activation( 'relu' ) )

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add( Dense( 1 ) )
model.add( Activation( 'sigmoid' ) )

model.compile( loss = 'binary_crossentropy',
               optimizer = 'rmsprop',
               )

hist=model.fit( X_train, y_train, batch_size = batch_size,
           nb_epoch = num_epochs, verbose = 1,
           validation_data = (X_test, y_test),
           )

basename = "model3"
suffix = datetime.datetime.now( ).strftime( "%y%m%d_%I%M%S" )
filename = "_".join( [ basename, suffix ] )



model.save_weights(os.path.join('./model_data/saved_weights',filename)+'.hdf5')


with open( os.path.join('./model_data/model_specs',filename )+'.config','w') as f:
	f.write( str(model.get_config( )) )
with open( os.path.join('./model_data_model_specs',filename )+'.hist','wb') as f:
	cPickle.dump(hist,f)
