import numpy as np
import pandas as pd
import random
import sys
from scipy.special import expit
from sklearn.preprocessing import LabelBinarizer, normalize
import math

class neural_net():

	def __init__(self,inp,hidden,outp,activation='sigmoid'):
		self.layers = [inp] + hidden + [outp]
		self.biases = [(2 * np.random.rand(i, 1) - 1) for i in self.layers[1:]]
		self.weights = [(2 * np.random.rand(self.layers[i+1], self.layers[i]) -1) for i in range(len(self.layers)-1)]
		self.activation = activation

	def activate(self,inp):
		val = 0
		if (self.activation == 'sigmoid'):
			val = expit(inp)
		if (self.activation == 'tanh'):
			val = np.tanh(inp)
		if (self.activation == 'relu'):
			if inp > 0:
				val = inp
		return val

	def derivate(self,inp):
		val = 0
		if (self.activation == 'sigmoid'):
			val = expit(inp)*(1-expit(inp))
		if (self.activation == 'tanh'):
			val = 1 - np.tanh(inp)**2
		if (self.activation == 'relu'):
			if inp <= 0:
				val = 0
			else:
				val = 1
		return val

	def feedforward(self,example):
		curr = example
		for i in range(len(self.layers)-1):
			w = self.weights[i]
			b = self.biases[i]			
			if(i == len(self.layers)-2):
				curr = expit(np.dot(w,curr)+b)
			else:
				curr = self.activate(np.dot(w,curr)+b)
		return curr

	def calcCost(self,a,y):
		return (np.linalg.norm(a-y)**2)/2

	def delta(self,z,a,y):
		return (a-y)*(expit(z)*(1-expit(z)))

	def get_node_vals(self,x,y):
		a = x
		acts = [x]
		zs = []
		for i in range(len(self.layers)-1):
			w = self.weights[i]
			b = self.biases[i]	
			z = np.dot(w, a)+b
			zs = zs + [z]
			a = self.activate(z)
			acts.append(a)
		return (acts,zs)

	def train(self,training_data,epochs,mini_batch_size,eta,evaluation_data=None,monitor_evaluation_cost=False,monitor_evaluation_accuracy=False,monitor_training_cost=False,monitor_training_accuracy=False):
		
		if evaluation_data:
			x_test = evaluation_data[0]
			y_test = evaluation_data[1]
			n_data = len(x_test)

		x_train = training_data[0]
		y_train = training_data[1]

		n = len(x_train)
		batches = int(n/mini_batch_size)
		
		evaluation_cost, evaluation_accuracy = [], []
		training_cost, training_accuracy = [], []
		prev_loss = 0
		k = 0
		eta_eff = eta

		for j in range(epochs):
			print (eta_eff)
			for i in range(batches-1):
				x_tr = x_train[i*mini_batch_size:(i+1)*mini_batch_size]
				y_tr = y_train[i*mini_batch_size:(i+1)*mini_batch_size]
				self.gradient_descent((x_tr,y_tr), eta_eff) # check this

			curr_loss = self.total_cost(list(zip(x_tr,y_tr)))
			if(curr_loss > prev_loss):
				k += 1
				eta_eff = eta/math.sqrt(k)
			prev_loss = curr_loss
				
			print ("Epoch ",j," training complete")

			if monitor_training_cost:
				cost = curr_loss
				training_cost.append(cost)
				print ("Cost on training data: ", cost)
			if monitor_training_accuracy:
				accuracy = self.accuracy(list(zip(x_train,y_train)))
				training_accuracy.append(accuracy)
				print ("Accuracy on training data: ",accuracy,"/",n)			
			if monitor_evaluation_cost:
				cost = self.total_cost(list(zip(x_test,y_test)))
				evaluation_cost.append(cost)
				print ("Cost on evaluation data:",cost)
			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(list(zip(x_test,y_test)))
				evaluation_accuracy.append(accuracy)
				print ("Accuracy on evaluation data: ",accuracy,"/",n_data)
		return evaluation_cost, evaluation_accuracy,training_cost, training_accuracy

	def gradient_descent(self, batch, eta):
		
		x_tr = batch[0]
		y_tr = batch[1]
		n = len(x_tr)
		tmp_b = [np.zeros(b.shape) for b in self.biases]
		tmp_w = [np.zeros(w.shape) for w in self.weights]

		for i in range(n):
			x = x_tr[i]
			y = y_tr[i]
			derivs = self.backpropagation(x, y)
			db = derivs[0]
			dw = derivs[1]
			for j in range(len(tmp_b)):
				tmp_b[j] = tmp_b[j] + db[j]
				tmp_w[j] = tmp_w[j] + dw[j]
		for i in range(len(tmp_b)):
			self.weights[i] = self.weights[i] - (2*eta/n)*tmp_w[i]
			self.biases[i] = self.biases[i] - (2*eta/n)*tmp_b[i]

	def backpropagation(self, x, y):

		node_vals = self.get_node_vals(x,y)
		acts = node_vals[0]
		zs = node_vals[1]

		db = [np.zeros(b.shape) for b in self.biases]
		dw = [np.zeros(w.shape) for w in self.weights]
		# Final Layer--always sigmoid
		delta = self.delta(zs[-1], acts[-1], y.reshape((46,1)))
		db[-1] = delta
		dw[-1] = np.dot(delta, acts[-2].transpose())

		for l in range(len(self.layers)):
			if(l > 1):
				delta = np.dot(self.weights[-l+1].transpose(), delta) * (self.derivate(zs[-l]))
				db[-l] = delta
				dw[-l] = np.dot(delta, acts[-l-1].transpose())

		derivs = (db,dw)
		return derivs

	def accuracy(self, data):
		results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
		return sum(int(x == y) for (x, y) in results)

	def total_cost(self, data):
		cost = 0.0
		for x, y in data:
			a = self.feedforward(x)
			cost += self.calcCost(a, y)/len(data)
		return cost

	def summate(self,delta,act):
		wt = []
		# print (delta.shape)
		# print (act.shape)
		for i in range(delta.shape[1]):
			tmpt = np.dot(delta[:,i:i+1],act[:,i:i+1].transpose())
			wt.append(tmpt)

		dsum = np.zeros((delta.shape[0],act.shape[0]))
		for t in wt:
			dsum += t

		return dsum


	def train2(self,training_data,epochs,mini_batch_size,eta,evaluation_data=None,monitor_evaluation_cost=False,monitor_evaluation_accuracy=False,monitor_training_cost=False,monitor_training_accuracy=False):
		
		if evaluation_data:
			x_test = evaluation_data[0]
			y_test = evaluation_data[1]
			n_data = len(x_test)

		x_train = training_data[0]
		y_train = training_data[1]

		n = (x_train.shape)[0]
		batches = int(n/mini_batch_size)
		
		evaluation_cost, evaluation_accuracy = [], []
		training_cost, training_accuracy = [], []
		prev_loss = 0
		k = 0
		eta_eff = eta

		for j in range(epochs):
			print (eta_eff)
			for i in range(batches-1):
				x_tr = x_train[i*mini_batch_size:(i+1)*mini_batch_size,:]
				y_tr = y_train[i*mini_batch_size:(i+1)*mini_batch_size,:]
				# print ("Iteration :", i)
				self.gradient_descent2((x_tr,y_tr), eta_eff) # check this

			# curr_loss = self.total_cost(list(zip(x_tr,y_tr)))
			# if(curr_loss > prev_loss):
			# 	k += 1
			# 	eta_eff = eta/math.sqrt(k)
			# prev_loss = curr_loss
				
			print ("Epoch ",j," training complete")

			# if monitor_training_cost:
			# 	cost = curr_loss
			# 	training_cost.append(cost)
			# 	print ("Cost on training data: ", cost)
			# if monitor_training_accuracy:
			# 	accuracy = self.accuracy(list(zip(x_train,y_train)))
			# 	training_accuracy.append(accuracy)
			# 	print ("Accuracy on training data: ",accuracy,"/",n)			
			# if monitor_evaluation_cost:
			# 	cost = self.total_cost(list(zip(x_test,y_test)))
			# 	evaluation_cost.append(cost)
			# 	print ("Cost on evaluation data:",cost)
			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(list(zip(x_test,y_test)))
				evaluation_accuracy.append(accuracy)
				print ("Accuracy on evaluation data: ",accuracy,"/",n_data)
		return evaluation_cost, evaluation_accuracy,training_cost, training_accuracy

	def gradient_descent2(self, batch, eta):
		
		x_tr = batch[0]
		y_tr = batch[1]
		n = (x_tr.shape)[0]
		tmp_b = [np.zeros(b.shape) for b in self.biases]
		tmp_w = [np.zeros(w.shape) for w in self.weights]

		derivs = self.backpropagation2(x_tr,y_tr)
		db = derivs[0]
		dw = derivs[1]
		# for i in range(n):
			# x = x_tr[i]
			# y = y_tr[i]
			# derivs = self.backpropagation(x, y)
		# db = derivs[0]
		# dw = derivs[1]
		for j in range(len(tmp_b)):
			tmp_b[j] = tmp_b[j] + db[j]
			tmp_w[j] = tmp_w[j] + dw[j]
		for i in range(len(tmp_b)):
			self.weights[i] = self.weights[i] - (2*eta/n)*tmp_w[i]
			self.biases[i] = self.biases[i] - (2*eta/n)*tmp_b[i]

	def backpropagation2(self, x, y):

		# node_vals = self.get_node_vals(x,y)
		# print (len(self.layers))
		a = np.transpose(x)
		acts = [a]
		zs = []
		for l in range(len(self.layers)-1):
			w = self.weights[l]
			b = self.biases[l]
			# print (l,": ",w.shape)
			# print (l,": ",a.shape)
			# print (l,": ",b.shape)
			z = np.dot(w, a)+b
			# print (l,": ",z.shape)
			zs = zs + [z]
			a = self.activate(z)
			acts.append(a)
		# return (acts,zs)

		# acts = node_vals[0]
		# zs = node_vals[1]

		db = [np.zeros(b.shape) for b in self.biases]
		dw = [np.zeros(w.shape) for w in self.weights]
		# Final Layer--always sigmoid
		delta = self.delta(zs[-1], acts[-1], y.transpose())
		db[-1] = np.sum(delta,axis=1)
		db[-1] = db[-1].reshape((db[-1].shape[0],1))
		# print ("DB last: ",db[-1].shape)

		# dw[-1] = np.dot(delta, acts[-2].transpose())
		dw[-1] = self.summate(delta,acts[-2])
		# print ("WT last: ",dw[-1].shape)

		for l in range(len(self.layers)):
			if(l > 1):
				delta = np.dot(self.weights[-l+1].transpose(), delta) * (self.derivate(zs[-l]))
				db[-l] = np.sum(delta,axis=1)
				db[-l] = db[-l].reshape((db[-l].shape[0],1))
				# print("layer :", (-l) , " : ",db[-l].shape)
				# dw[-l] = np.dot(delta, acts[-l-1].transpose())
				dw[-l] = self.summate(delta,acts[-l-1])
				# print ("layer :", (-l), " :",dw[-l].shape)

		derivs = (db,dw)
		return derivs

# def sigmoid_prime(z):
# 	return sigmoid(z)*(1-sigmoid(z))

train = pd.read_csv("../dummy_evaluation/data/devnagri_train.csv",header=None).values
test = pd.read_csv("../dummy_evaluation/data/devnagri_test.csv",header=None).values
# print (train.shape)
# print (test.shape)
x_train = normalize(train[:,1:])
x_test = normalize(test[:,1:])
y_train = train[:,:1]
y_test = pd.read_csv('../dummy_evaluation/data/devnagri_target_labels.txt',header=None).values
one_hot = (LabelBinarizer()).fit(y_train)
y_train = one_hot.transform(y_train)
y_test = one_hot.transform(y_test)

x_tr = [x.reshape((1024,1)) for x in x_train]
x_ts = [x.reshape((1024,1)) for x in x_test]
y_tr = [y.reshape((46,1)) for y in y_train]
y_ts = [y.reshape((46,1)) for y in y_test]

train_data = (x_train,y_train)
test_data = (x_ts,y_ts)

net = neural_net(1024,[100],46,activation = 'sigmoid')
net.train2(train_data,51,512,0.5,evaluation_data=test_data,monitor_evaluation_accuracy=True,monitor_evaluation_cost=True,monitor_training_accuracy=True,monitor_training_cost=True)