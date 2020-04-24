#!/usr/bin/env python
import numpy as np

class GradientDescent:
	
	def cal_cost(self,theta,X,y):
		m = len(y)
		predictions = X.dot(theta)
		cost = (1/2*m) * np.sum(np.square(predictions-y))
		return cost

	def __initialize(self,x,y,iterations):
		m = len( y ) #total examples
		X = np.c_[ np.ones((m,1)), x ] #add bias
		n = X.shape[1] #total features
		theta = np.random.randn( n, 1 ) #weights initialized
		if type( y[0] ) == int:
			y = np.array( [ [y_i] for y_i in y ] )
		cost_history = np.zeros(iterations)
		theta_history = np.zeros( (iterations, n) )

		return X,y,m,n,theta,cost_history,theta_history

	def __gradient(self,x, y, theta):
		prediction = np.dot(x,theta)
		return x.T.dot((prediction - y))

	def predict(self, x, theta):
		x_b = np.c_[ np.ones((x.shape[0],1)), x ]
		return x_b.dot(theta)

	def gradient_descent(self, x, y, learning_rate=0.01, iterations=100 ):
		X,y,m,n,theta,cost_history,theta_history = self.__initialize(x,y,iterations)
		
		for it in range(iterations):
			theta = theta -(1/m)*learning_rate*self.__gradient(X,y,theta)
			theta_history[it,:] =theta.T
			cost_history[it]  = self.cal_cost(theta,X,y)
		
		return theta, cost_history, theta_history
			
	def stochastic_gradient_descent(self, x, y, learning_rate=0.01, iterations=10 ):
		X,y,m,n,theta,cost_history,theta_history = self.__initialize(x,y,iterations)

		for it in range(iterations):
			cost =0.0
			for i in range( m ): #for each examples
				rand_ind = np.random.randint(0,m) #pick any example randomly
				x_i = X[rand_ind,:].reshape(1, n) #pick feature of example
				y_i = np.array( y[rand_ind] ).reshape(1,1)    #pick output
				theta = theta - (1/m)*learning_rate*self.__gradient(x_i, y_i, theta)
				cost += self.cal_cost(theta, x_i, y_i)
			
			cost_history[it] = cost
			theta_history[it,:] = theta.T
		
		return theta, cost_history, theta_history
		
	def minibatch_gradient_descent(self, x, y, learning_rate=0.01, iterations=10, batch_size =20 ):
		X,y,m,n,theta,cost_history,theta_history = self.__initialize(x,y,iterations)

		n_batches = int(m/batch_size)
		for it in range(iterations):
			cost =0.0
			indices = np.random.permutation(m)
			X = X[indices]
			y = y[indices]
			for i in range(0,m,batch_size):
				X_i = X[i:i+batch_size]
				y_i = y[i:i+batch_size]
				theta = theta - (1/m)*learning_rate*self.__gradient(X_i, y_i, theta)
				cost += self.cal_cost(theta,X_i,y_i)
			
			cost_history[it]  = cost
			theta_history[it,:] = theta.T
		
		return theta, cost_history, theta_history