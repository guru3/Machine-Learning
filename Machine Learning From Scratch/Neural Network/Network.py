import numpy as np

#TODO - add more activation functions

class Sigmoid:
	def activation( self, x ):
		return 1.0/(1 + np.exp(-x));

	def gradient( self, x ):
		activationV = self.activation(x);
		return activationV * ( 1 - activationV );
        
class NeuralNetwork:

	def __init__( self, layers ):
		#fully connected network - we just need to know weights and biases at each location
		self.totalLayers = len(layers); #first is input layer, last one is output layer
		self.layerSizes = layers;
		self.threshold = 0.5 #threshold to decide
		self.activationFunction = Sigmoid(); 
		self.resetNetwork();

	def initializeWeights( self ):
		#TODO - support many different initializations
		assert self.totalLayers >= 2 #atleast one for input for another for output
		self.weights = [ np.random.rand(x,y) for (x,y) in zip(self.layerSizes[1:], self.layerSizes[0:-1]) ];
		self.biases = [ np.random.rand(x,1) for x in self.layerSizes[1:] ];

	def resetNetwork( self ):
		self.initializeWeights();

	def train( self, x_train, y_train, totalEpochs = 10, learningRate = 0.01 ):
		self.resetNetwork() #let's reset the network
		self.assertInputOutput( x_train, y_train);
		totalExamples = x_train.shape[1] #x = [ #features x #examples ], y = [ #totalOutputs, #examples ]
		for epochNo in range(totalEpochs):
			#calculate gradients using backprop
			#use gradient descent to update weights/biases
			#TODO - support different gradient descent optimization techniques
			pass

	def assertInputOutput( self, x, y ):
		assert( x.shape[0] == self.layerSizes[0] )
		assert( x_shape[1] == y_shape[1] ) #total examples

	def forward_propagation( self, x ):
		output = x;
		for i in range(len(self.weights)):
			dotProduct = np.dot( self.weights[i], output )
			assert( dotProduct.shape == (self.weights[i].shape[0],output.shape[1]))
			assert( dotProduct.shape[0] == self.biases[i].shape[0] )
			dotProduct = np.add( dotProduct, self.biases[i])
			output = self.activationFunc.activation( dotProduct )
			assert( output.shape == dotProduct.shape )
		return np.argmax( output, axis=0 )

	def evaluate( self, x_test, y_test ):
		y_pred = self.forward_propagation(x_test)
		count = np.sum( [ (int)(y_pred[i]==y_test[i]) for i in range(len(y_pred)) ] )
		return count;