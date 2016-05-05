from keras.engine.topology import Layer
from keras import initializations
import numpy as np
import keras.backend as K





class DistanceMetric( Layer ):
	def __init__( self, alpha_init = 0.2, beta_init = 5.0,
	              weights = None, **kwargs ):
		self.supports_masking = False
		# self.alpha_init = K.cast_to_floatx( alpha_init )
		# self.beta_init = K.cast_to_floatx( beta_init )
		self.initial_weights = weights
		super( DistanceMetric, self ).__init__( **kwargs )

	def build( self, input_shape ):
		input_shape = input_shape[ 1: ]
		# self.alphas = K.variable( self.alpha_init * np.ones( input_shape ),
		#                           name = '{}_alphas'.format( self.name ) )
		# self.betas = K.variable( self.beta_init * np.ones( input_shape ),
		#                          name = '{}_betas'.format( self.name ) )
		# self.trainable_weights = [ self.alphas, self.betas ]

		if self.initial_weights is not None:
			self.set_weights( self.initial_weights )
			del self.initial_weights

	def call(self, x, mask=None):
		DistanceMetric.__call__(x,mask)

	def __call__( self, x, mask = None ):
		dotProd=K.dot(x[:len(x)/2],x[len(x)/2:])
		l2norm = K.l2_normalize(dotProd)
		return l2norm

	def get_config( self ):
		config ={}
		# config = { 'alpha_init': self.alpha_init,
		#            'beta_init' : self.beta_init }
		base_config = super( DistanceMetric, self ).get_config( )
		return dict( list( base_config.items( ) ) + list( config.items( ) ) )
