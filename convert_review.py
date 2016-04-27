from __future__ import print_function
import numpy
import os
import cPickle
import multiprocessing
import modelParameters
import random
from functools import partial
from preprocess import (generate_word_list, generate_char_list,
                        generate_one_hot_maps, sentiment2reviews_map)

DESIGN_MATRIX_PATH_WORD = '../model_data/designMatrix_w.pickle'
TARGET_VECTOR_PATH_WORD = '../model_data/targetVect_w.pickle'

DESIGN_MATRIX_PATH_CHAR = '../model_data/designMatrix_c.pickle'
TARGET_VECTOR_PATH_CHAR = '../model_data/targetVect_c.pickle'

DEV_DESIGN_MATRIX_PATH_WORD = '../model_data/dev_designMatrix_w.pickle'
DEV_TARGET_VECTOR_PATH_WORD = '../model_data/dev_targetVect_w.pickle'

DEV_DESIGN_MATRIX_PATH_CHAR = '../model_data/dev_designMatrix_c.pickle'
DEV_TARGET_VECTOR_PATH_CHAR = '../model_data/dev_targetVect_c.pickle'

TEST_SET_DATA_PATH_WORD = '../model_data/test_set_data_w.pickle'
TEST_SET_DATA_PATH_CHAR = '../model_data/test_set_data_c.pickle'
TEST_SET_ID_VECTOR='../model_data/test_set_ID_vect.pickle'



def to_onehot_vector( reviewObject, vocab_size,one_hot_maps,
                      use_words,skip_top=0, maxlen=None, **kwargs ):
	rating, review = reviewObject




	if use_words:
		MAXlen = maxlen if maxlen is not None else modelParameters.MaxLen_w
		vector_of_onehots = numpy.zeros( (1, maxlen) )
		vector_of_onehots += modelParameters.UNK_INDEX
		for indx, word in enumerate( generate_word_list( review )[ :MAXlen ] ):
			vector_of_onehots[ 0, indx ] = one_hot_maps[ word ]

	else:
		MAXlen = maxlen if maxlen is not None else modelParameters.MaxLen_c
		vector_of_onehots = numpy.zeros( (1, maxlen) )
		vector_of_onehots += modelParameters.UNK_INDEX
		for indx, char in enumerate( generate_char_list( review )[ :MAXlen ] ):
			vector_of_onehots[ 0, indx ] = one_hot_maps[ char ]

	return (vector_of_onehots, rating)


def build_design_matrix( vocab_size, use_words,
                         skip_top = 0, maxlen = None,dev_split=None,
                         verbose = True, **kwargs ):

	if kwargs.get('test_data',None):

		if use_words and os.path.isfile(TEST_SET_DATA_PATH_WORD):
			print( "word TEST design matrix found, loading pickle" )
			with open( TEST_SET_DATA_PATH_WORD, 'rb' ) as f:
				designMatrix = cPickle.load( f )
			with open( TEST_SET_ID_VECTOR, 'rb' ) as f:
				targets = cPickle.load( f )

			return (designMatrix, targets)

		elif not use_words and os.path.isfile(TEST_SET_DATA_PATH_CHAR):
			print( "char TEST design matrix found, loading pickle" )
			with open( TEST_SET_DATA_PATH_CHAR, 'rb' ) as f:
				designMatrix = cPickle.load( f )
			with open( TEST_SET_ID_VECTOR, 'rb' ) as f:
				targets = cPickle.load( f )

			return (designMatrix,targets)




	else:

		if use_words and os.path.isfile( DESIGN_MATRIX_PATH_WORD ):
			print( "word TRAINING design matrix found, loading pickle" )
			with open( DESIGN_MATRIX_PATH_WORD, 'rb' ) as f:
				designMatrix = cPickle.load( f )
			with open( TARGET_VECTOR_PATH_WORD, 'rb' ) as f:
				targets = cPickle.load( f )
			with open(DEV_DESIGN_MATRIX_PATH_WORD,'rb') as f:
				dev_designMatrix=cPickle.load(f)
			with open(DEV_TARGET_VECTOR_PATH_WORD,'rb') as f:
				dev_targets = cPickle.load(f)

			return ((designMatrix, targets),
			        (dev_designMatrix,dev_targets))

		elif not use_words and  os.path.isfile( DESIGN_MATRIX_PATH_CHAR ):

			print( "char TRAINING design matrix found, loading pickle" )
			with open( DESIGN_MATRIX_PATH_CHAR, 'rb' ) as f:
				designMatrix = cPickle.load( f )
			with open( TARGET_VECTOR_PATH_CHAR, 'rb' ) as f:
				targets = cPickle.load( f )
			with open( DEV_DESIGN_MATRIX_PATH_CHAR, 'rb' ) as f:
				dev_designMatrix = cPickle.load( f )
			with open( DEV_TARGET_VECTOR_PATH_CHAR, 'rb' ) as f:
				dev_targets = cPickle.load( f )

			return ((designMatrix, targets),
			        (dev_designMatrix, dev_targets))





	if verbose:
		print( "pickled data not found, building it..." )

	testing_phase = kwargs.get( 'test_data', None )

	one_hots = generate_one_hot_maps( vocab_size, skip_top, use_words )


	if testing_phase:
		print( "building test data objects" )
		print( "test data has no targets;\n"
		       "so the targets vector will contain ID of review at that index" )

		review_iterator = list()
		for review_file in os.listdir('./testing_data/test/'):
			with open('./testing_data/test/'+review_file) as f:
				#review id and review text in tuple
				review_iterator.append((review_file[:-4],f.read()))

		if use_words:
			print( "building TEST word design matrix" )
			designMatrix = numpy.zeros( (modelParameters.testingCount,
			                             (maxlen if maxlen is not None
			                              else modelParameters.MaxLen_w)) )

		else:
			print( "building TEST char design matrix" )
			designMatrix = numpy.zeros( (modelParameters.testingCount,
			                             (maxlen if maxlen is not None
			                              else modelParameters.MaxLen_c)) )

		##for test data targets vector will hold review IDs; not ratings
		targets = numpy.zeros( (modelParameters.testingCount, 1) )



	else:

		sentiment_reviews = sentiment2reviews_map( )


		if use_words:
			print( "building TRAINING word design matrix" )
			designMatrix = numpy.zeros( (modelParameters.trainingCount,
			                             (maxlen if maxlen is not None
			                              else modelParameters.MaxLen_w)) )
		else:
			print( "building TRAINING char design matrix" )
			designMatrix = numpy.zeros( (modelParameters.trainingCount,
			                             (maxlen if maxlen is not None
			                              else modelParameters.MaxLen_c)) )

		targets = numpy.zeros( (modelParameters.trainingCount, 1) )

		review_iterator = list( )

		for label, file_map in sentiment_reviews.iteritems( ):
			for stars, reviewList in file_map.iteritems( ):
				for review in reviewList:
					review_iterator.append( (stars, review) )


	##now in common area where both test and training phase will execute
	MAXlen = (maxlen if maxlen is not None else
	          modelParameters.MaxLen_w if use_words else
	          modelParameters.MaxLen_c)


	func_to_one_hot = partial( to_onehot_vector,
	                           one_hot_maps=one_hots,
	                           vocab_size = vocab_size,
	                           use_words = use_words,
	                           skip_top = skip_top,
	                           maxlen = MAXlen,
	                           )

	workers = multiprocessing.Pool( processes = 8 )
	results = workers.map( func_to_one_hot,
	                       review_iterator )
	workers.close( )
	workers.join( )


	if dev_split is not None:
		print("creating dev set")
		random.shuffle(results)

		split = int((float(dev_split)/100)*modelParameters.trainingCount)
		dev_set = results[:split]
		results = results[split:]

		dev_designMatrix = numpy.zeros((len(dev_set),MAXlen))
		dev_targets = numpy.zeros((len(dev_set),1))

		designMatrix = numpy.resize(designMatrix,(len(results),MAXlen))
		targets = numpy.resize(targets,(len(results),1))

		for idx, (vector, rating) in enumerate( dev_set ):
			dev_designMatrix[ idx, : ] = vector

			dev_targets[ idx, 0 ] = rating >= 7



	for idx, (vector, rating) in enumerate( results ):
		designMatrix[ idx, : ] = vector
		if testing_phase:
			targets[ idx, 0 ] = rating
		else:
			targets[ idx, 0 ] = rating >= 7

	print( "finished building data design matrix, now pickling" )

	if testing_phase is not None:
		##test ID vector is same for both char and word
		with open( TEST_SET_ID_VECTOR, 'wb' ) as f:
			cPickle.dump( targets, f )

		if use_words:
			with open( TEST_SET_DATA_PATH_WORD, 'wb' ) as f:
				cPickle.dump( designMatrix, f )

		else:
			with open( TEST_SET_DATA_PATH_CHAR, 'wb' ) as f:
				cPickle.dump( designMatrix, f )


	else:
		if use_words:
			with open( DESIGN_MATRIX_PATH_WORD, 'wb' ) as f:
				cPickle.dump( designMatrix, f )
			with open( TARGET_VECTOR_PATH_WORD, 'wb' ) as f:
				cPickle.dump( targets, f )

			if dev_split is not None:
				with open( DEV_DESIGN_MATRIX_PATH_WORD, 'wb' ) as f:
					cPickle.dump( dev_designMatrix, f )
				with open( DEV_TARGET_VECTOR_PATH_WORD, 'wb' ) as f:
					cPickle.dump( dev_targets, f )
		else:
			pass
			# with open( DESIGN_MATRIX_PATH_CHAR, 'wb' ) as f:
			# 	cPickle.dump( designMatrix, f )
			# with open( TARGET_VECTOR_PATH_CHAR, 'wb' ) as f:
			# 	cPickle.dump( targets, f )
			#
			# if dev_split is not None:
			# 	with open( DEV_DESIGN_MATRIX_PATH_CHAR, 'wb' ) as f:
			# 		cPickle.dump( dev_designMatrix, f )
			# 	with open( DEV_TARGET_VECTOR_PATH_CHAR, 'wb' ) as f:
			# 		cPickle.dump( dev_targets, f )

	if dev_split is not None:
		return ((designMatrix,targets),(dev_designMatrix,dev_targets))
	else:
		return (designMatrix, targets)



def get_testing_data(vocab_size,use_words):
	return build_design_matrix( vocab_size = vocab_size,
	                              use_words = use_words,
	                              test_data = True )

if __name__ == '__main__':

	USEWORDS = True

	if USEWORDS:
		V = modelParameters.VocabSize_w
	else:
		V = modelParameters.VocabSize_c

	SKP = modelParameters.skip_top

	DEVSPLIT=14


	(x, y) = build_design_matrix( vocab_size = V,
	                              use_words = USEWORDS,
	                              skip_top = SKP,
	                              dev_split = DEVSPLIT)
