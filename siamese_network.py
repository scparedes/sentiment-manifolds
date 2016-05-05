
from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers.core import (Dense, Dropout,
                               Activation, Flatten)
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import theano
import cPickle
from keras.regularizers import l2, l1, l1l2
from convert_review import build_design_matrix
import keras.backend as K
import modelParameters
import os
import datetime
from activations import DistanceMetric

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

def contrastiveLoss(Xl,Xr,y):
	return y*K.l2_normalize(Xl,Xr) + (1-y)*K.max(10,10-K.l2_normalize(Xl,Xr))


batch_size = 80

num_filters1 = 1300
filter_length1 = 2
stride_len1 = 1
pool_len1 = 2

num_filters2 = 800
filter_length2 = 3
stride_len2 = 1
pool_len2 = 2

num_filters3 = 500
filter_length3 = 4
stride_len3 = 1
pool_len3 = 2


num_filters4 = 300
filter_length4 = 5
stride_len4 = 1
pool_len4 = 2


embedding_dims = 200

hidden_dims = 100
num_epochs = 5

def train_siamese_model():

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


	#LEFT and RIGHT branches of siamese network
	modelL = Sequential( )
	modelR = Sequential( )


	# we start off with an efficient embedding layer which maps
	# our vocab indices into embedding_dims dimensions
	modelL.add( Embedding( VocabSize, embedding_dims, input_length = maxReviewLen ) )
	modelR.add( Embedding( VocabSize, embedding_dims, input_length = maxReviewLen ) )

	modelL.add( Dropout( 0.10 ) )
	modelR.add( Dropout( 0.10 ) )
	###init changed from uniform to glorot_norm
	modelL.add( Convolution1D( nb_filter = num_filters1,
	                          filter_length = filter_length1,
	                          border_mode = 'valid',
	                          activation = 'relu',
	                          subsample_length = stride_len1,
	                          init = 'uniform'
	                          ) )

	modelR.add( Convolution1D( nb_filter = num_filters1,
	                           filter_length = filter_length1,
	                           border_mode = 'valid',
	                           activation = 'relu',
	                           subsample_length = stride_len1,
	                           init = 'uniform'
	                           ) )


	# input_length=maxReviewLen, input_dim=VocabSize

	modelL.add( Dropout( 0.25 ) )
	modelR.add( Dropout( 0.25 ) )


	modelL.add( MaxPooling1D( pool_length = 2 ) )
	modelR.add( MaxPooling1D( pool_length = 2 ) )




	modelL.add( Convolution1D( nb_filter = num_filters2,
	                          filter_length = filter_length2,
	                          border_mode = 'valid',
	                          activation = 'relu',
	                          subsample_length = stride_len2,
	                          init = 'uniform'

	                          ) )

	modelR.add( Convolution1D( nb_filter = num_filters2,
	                           filter_length = filter_length2,
	                           border_mode = 'valid',
	                           activation = 'relu',
	                           subsample_length = stride_len2,
	                           init = 'uniform'

	                           ) )



	modelL.add( Dropout( 0.40 ) )
	modelR.add( Dropout( 0.40 ) )

	# we use standard max pooling (halving the output of the previous layer):
	modelL.add( MaxPooling1D( pool_length = 2 ) )
	modelR.add( MaxPooling1D( pool_length = 2 ) )




	modelL.add( Convolution1D( nb_filter = num_filters3,
	                          filter_length = filter_length3,
	                          border_mode = 'valid',
	                          activation = 'relu',
	                          subsample_length = stride_len3,
	                          init = 'uniform'

	                          ) )

	modelR.add( Convolution1D( nb_filter = num_filters3,
	                           filter_length = filter_length3,
	                           border_mode = 'valid',
	                           activation = 'relu',
	                           subsample_length = stride_len3,
	                           init = 'uniform'

	                           ) )







	modelL.add( Dropout( 0.30 ) )
	modelR.add( Dropout( 0.30 ) )




	modelL.add( MaxPooling1D( pool_length = 2 ) )

	modelR.add( MaxPooling1D( pool_length = 2 ) )




	modelL.add( Convolution1D( nb_filter = num_filters4,
	                          filter_length = filter_length4,
	                          border_mode = 'valid',
	                          activation = 'relu',
	                          subsample_length = stride_len4,
	                          init = 'uniform'

	                          ) )

	modelR.add( Convolution1D( nb_filter = num_filters4,
	                           filter_length = filter_length4,
	                           border_mode = 'valid',
	                           activation = 'relu',
	                           subsample_length = stride_len4,
	                           init = 'uniform'

	                           ) )



	modelL.add( Dropout( 0.25 ) )
	modelR.add( Dropout( 0.25 ) )

	modelL.add( MaxPooling1D( pool_length = pool_len4 ) )
	modelR.add( MaxPooling1D( pool_length = pool_len4 ) )






	# We flatten the output of the conv layer,
	# so that we can add a vanilla dense layer:
	modelL.add( Flatten( ) )
	modelR.add( Flatten( ) )

	# We add a vanilla hidden layer:
	modelL.add( Dense( hidden_dims ) )
	modelL.add( Activation( 'relu' ) )

	modelR.add( Dense( hidden_dims ) )
	modelR.add( Activation( 'relu' ) )



	merged_vector = merge([modelL, modelR], mode='concat', concat_axis=-1)


	Gw=Dense(1, activation=DistanceMetric)(merged_vector)


	model.compile( loss = contrastiveLoss,
	               optimizer = 'rmsprop',
	               )
