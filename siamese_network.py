from __future__ import print_function
import numpy as np



from keras.models import Sequential
from keras.layers.core import (Dense, Dropout,
                               Activation, Flatten,
                               Merge)
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import (Convolution1D,
                                        MaxPooling1D)

import theano
import modelParameters


USEWORDS=False

if USEWORDS:
	VocabSize = modelParameters.VocabSize_w
	maxReviewLen = modelParameters.MaxLen_w
	skipTop = modelParameters.skip_top
else:
	VocabSize = modelParameters.VocabSize_c
	maxReviewLen = modelParameters.MaxLen_c
	skipTop =0


DEVSPLIT=14
batch_size = 32

embedding_dims = 601    #dimension of word vectors: D


num_filters1 = 1530
filter_length1 = 2
stride_len1 = 1
pool_len1 = 2

num_filters2 = 1003
filter_length2 = 3
stride_len2 = 2
pool_len2 = 2

num_filters3 = 531
filter_length3 = 3
stride_len3 = 2
pool_len3 = 2


hidden_dims1 = 999
num_epochs = 1





modelL = Sequential()
modelR = Sequential()



# modelL.add( Embedding( VocabSize, embedding_dims, input_length=maxReviewLen ) )
# modelL.add(Dropout(0.25))
#
# modelR.add( Embedding( VocabSize, embedding_dims, input_length=maxReviewLen ) )
# modelR.add(Dropout(0.25))




#while using conv as 1st layer, input_length and input_dim
#must be added, but if using embedding b4 then remove those params

modelL.add(Convolution1D( nb_filter=num_filters1,
                          filter_length=filter_length1,
                          border_mode='same',
                          activation='relu',
                          subsample_length=stride_len1,
	                      input_length = maxReviewLen,
                          input_dim = VocabSize))

modelR.add(Convolution1D( nb_filter=num_filters1,
                          filter_length=filter_length1,
                          border_mode='same',
                          activation='relu',
                          subsample_length=stride_len1,
                          input_length = maxReviewLen,
                          input_dim = VocabSize
                          ))

modelL.add(Dropout(0.25))
modelR.add(Dropout(0.25))




modelL.add(MaxPooling1D(pool_length=pool_len1))
modelR.add(MaxPooling1D(pool_length=pool_len1))




modelL.add(Convolution1D( nb_filter=num_filters2,
                          filter_length=filter_length2,
                          border_mode='same',
                          activation='relu',
                          subsample_length=stride_len2 ))

modelR.add(Convolution1D( nb_filter=num_filters2,
                          filter_length=filter_length2,
                          border_mode='same',
                          activation='relu',
                          subsample_length=stride_len2 ))


modelL.add(MaxPooling1D(pool_length=pool_len2))
modelR.add(MaxPooling1D(pool_length=pool_len2))





modelL.add(Convolution1D( nb_filter=num_filters3,
                          filter_length=filter_length3,
                          border_mode='same',
                          activation='relu',
                          subsample_length=stride_len3 ))

modelR.add(Convolution1D( nb_filter=num_filters3,
                          filter_length=filter_length3,
                          border_mode='same',
                          activation='relu',
                          subsample_length=stride_len3 ))


modelL.add(MaxPooling1D(pool_length=pool_len3))
modelR.add(MaxPooling1D(pool_length=pool_len3))





modelL.add(Flatten())
modelR.add(Flatten())



modelL.add( Dense( hidden_dims1 ) )
modelL.add(Dropout(0.25))
modelL.add(Activation('relu'))


modelR.add( Dense( hidden_dims1 ) )
modelR.add(Dropout(0.25))
modelR.add(Activation('relu'))


mergeModel=Sequential()

mergeModel.add(Merge([modelL, modelR], mode='', ))




# modelL.add(Dense(1))
# modelL.add(Activation('sigmoid'))
#
# modelR.add(Dense(1))
# modelR.add(Activation('sigmoid'))
