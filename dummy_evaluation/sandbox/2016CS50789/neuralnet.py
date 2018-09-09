import numpy as np
import pandas as pd
import random
import sys
from scipy.special import expit
from sklearn.preprocessing import LabelBinarizer, normalize, StandardScaler
from sklearn.metrics import accuracy_score
import math

class neural_net():

	def __init__(self,inp,hidden,outp,activation='sigmoid'):
		self.layers = [inp] + hidden + [outp]
		self.width = len(self.layers)
		self.biases = [(2 * np.random.rand(i, 1) - 1) for i in self.layers[1:]]
		self.weights = [(2 * np.random.rand(self.layers[i+1], self.layers[i]) -1) for i in range(self.width-1)]
		self.activation = activation

	def activate(self,inp):
		val = 0
		if (self.activation == 'sigmoid'):
			val = expit(inp)
		if (self.activation == 'tanh'):
			val = np.tanh(inp)
		if (self.activation == 'relu'):
			val = np.maximum(inp,0)
		return val

	def derivate(self,inp):
		val = 0
		if (self.activation == 'sigmoid'):
			val = expit(inp)*(1-expit(inp))
		if (self.activation == 'tanh'):
			val = 1 - np.tanh(inp)**2
		if (self.activation == 'relu'):
			val = inp
			val[inp<=0] = 0
			val[inp>0] = 1
			# val = inp
		return val

	def feedforward(self,example):
		curr = example
		for i in range(self.width-1):
			w = self.weights[i]
			b = self.biases[i]			
			if(i == len(self.layers)-2):
				curr = expit(np.dot(w,curr)+b)
			else:
				curr = self.activate(np.dot(w,curr)+b)
		return curr

	def accuracy(self, Xdata, Ydata):
		return accuracy_score(self.predict(Xdata),Ydata)

	def predict(self,data):
		a = np.transpose(data)
		acts = [a]
		zs = []
		for l in range(self.width-1):
			z = np.dot(self.weights[l], a)+self.biases[l]
			zs = zs + [z]
			if(l < self.width - 2):
				a = self.activate(z)
				acts.append(a)
		# using sigmoid for output layer
		a = expit(zs[-1])
		acts.append(a)
		return np.argmax(acts[-1].transpose(),axis=1)

	def summate(self,delta,act):
		wt = []
		for i in range(delta.shape[1]):
			tmpt = np.dot(delta[:,i:i+1],act[:,i:i+1].transpose())
			wt.append(tmpt)

		dsum = np.zeros((delta.shape[0],act.shape[0]))
		for t in wt:
			dsum += t

		return dsum


	def train(self,training_data,epochs,batch_size,eta,evaluation_data=None,monitor_evaluation_accuracy=False,monitor_training_cost=False):
		
		if evaluation_data:
			x_test = evaluation_data[0]
			y_test = evaluation_data[1]
			n_data = len(x_test)

		self.X = training_data[0]
		self.Y = training_data[1]
		self.data_size = (self.X.shape[0])
		n = self.data_size
		batches = int(n/batch_size)

		prev_loss = [0,500]
		k = 0
		eta_eff = eta
		parity = 0

		for j in range(epochs):
			print (eta_eff)
			curr_loss = 0
			for i in range(batches-1):
				if (i%2 == parity):
					curr_loss += self.gradient_descent(i,batch_size,eta_eff)

			curr_loss /= n
			if(curr_loss > prev_loss[parity]):
				k += 1
				eta_eff = eta/math.sqrt(k)
			prev_loss[parity] = curr_loss
			parity = (parity+1)%2
				
			print ("Epoch ",j," training complete: ",curr_loss)

			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(x_test,y_test)
				print ("Accuracy on evaluation data: ",accuracy)

	def gradient_descent(self, st, batch_size,eta):

		tmp_b = [np.zeros(b.shape) for b in self.biases]
		tmp_w = [np.zeros(w.shape) for w in self.weights]

		derivs,err = self.backpropagation(st,batch_size)
		db = derivs[0]
		dw = derivs[1]

		for j in range(len(tmp_b)):
			tmp_b[j] = tmp_b[j] + db[j]
			tmp_w[j] = tmp_w[j] + dw[j]
		for i in range(len(tmp_b)):
			self.weights[i] = self.weights[i] - (2*eta/batch_size)*tmp_w[i]
			self.biases[i] = self.biases[i] - (2*eta/batch_size)*tmp_b[i]

		return err

	def backpropagation(self, st, batch_size):

		a = np.transpose(self.X[st*batch_size:(st+1)*batch_size,:])
		acts = [a]
		zs = []
		for l in range(self.width-1):
			z = np.dot(self.weights[l], a)+self.biases[l]
			zs = zs + [z]
			if(l < self.width - 2):
				a = self.activate(z)
				acts.append(a)
		# using sigmoid for output layer
		a = expit(zs[-1])
		acts.append(a)

		db = [np.zeros(b.shape) for b in self.biases]
		dw = [np.zeros(w.shape) for w in self.weights]

		diff = acts[-1] - (self.Y[st*batch_size:(st+1)*batch_size,:]).transpose()
		# Using sigmoid for output layer
		delta = diff*(expit(zs[-1]))*(1-expit(zs[-1]))
		nor = np.sum((np.linalg.norm(diff,axis=1))**2)

		db[-1] = (np.sum(delta,axis=1)).reshape((db[-1].shape[0],1))
		dw[-1] = self.summate(delta,acts[-2])

		for l in range(len(self.layers)):
			if(l > 1):
				delta = np.dot(self.weights[-l+1].transpose(), delta) * (self.derivate(zs[-l]))
				db[-l] = (np.sum(delta,axis=1)).reshape((db[-l].shape[0],1))
				dw[-l] = self.summate(delta,acts[-l-1])

		derivs = (db,dw)
		return derivs,nor

def remove_padding(data):
	a = ()
	for i in range(2,30):
		a = a + ((data[:,(i*32)+2:(i+1)*32 - 2]),)
	return (np.column_stack(a))

# train = pd.read_csv("../../col341_a2_data/devnagri_train.csv",header=None).values

# test = pd.read_csv("../dummy_evaluation/data/devnagri_test.csv",header=None).values
# # print (train.shape)
# # print (test.shape)
# x_train = normalize(train[:,1:])
# x_test = normalize(test[:,1:])
# SS = StandardScaler()
# x_train = SS.fit_transform(train[:,1:].astype(float))
# print (x_train.shape)
# x_test = SS.transform(test[:,1:])
# y_train = train[:,:1]
# y_test = pd.read_csv('../dummy_evaluation/data/devnagri_target_labels.txt',header=None).values
# one_hot = (LabelBinarizer()).fit(y_train)
# y_train = one_hot.transform(y_train)
# # y_test = one_hot.transform(y_test)
# x_train = remove_padding(x_train)
# x_test = remove_padding(x_test)

# # x_tr = [x.reshape((784,1)) for x in x_train]
# # x_ts = [x.reshape((784,1)) for x in x_test]
# # y_tr = [y.reshape((46,1)) for y in y_train]
# # y_ts = [y.reshape((46,1)) for y in y_test]

# train_data = (x_train,y_train)
# test_data = (x_test,y_test)

# # net = neural_net(784,[100],46,activation = 'tanh')
# # net.train(train_data,51,512,2,evaluation_data=test_data,monitor_evaluation_accuracy=True,monitor_training_cost=True)
# # pred = net.predict(x_test)

# net = neural_net(x_train.shape[1],[100,100],y_train.shape[1],activation='sigmoid')
# 	# net.train(train_data,50,bs,lr)
# net.train(train_data,100,128,2,evaluation_data=test_data,monitor_evaluation_accuracy=True)

arguments = sys.argv
part = arguments[1]
tr = arguments[2]
ts = arguments[3]
outfile = arguments[4]

SS = StandardScaler()
train = pd.read_csv(tr,header=None).values
test = pd.read_csv(ts,header=None).values
x_train = SS.fit_transform(remove_padding(normalize(train[:,1:])).astype(float))
x_test = SS.transform(remove_padding(normalize(test[:,1:])).astype(float))

y_train = train[:,:1]
one_hot = (LabelBinarizer()).fit(y_train)
y_train = one_hot.transform(y_train)

train_data = (x_train,y_train)

#512 0.01 "sigmoid" 100
if part == 'a':
	bs = int(arguments[5])
	lr = float(arguments[6])
	ac = arguments[7]
	hl = []
	for l in range(8,len(arguments)):
		hl.append(int(arguments[l]))
	net = neural_net(x_train.shape[1],hl,y_train.shape[1],activation=ac)
	net.train(train_data,75,bs,lr)
	pred = net.predict(x_test)
	np.savetxt(outfile,pred,fmt="%i")

if part =='b':
	net = neural_net(x_train.shape[1],[100,100],y_train.shape[1],activation='sigmoid')
	net.train(train_data,90,128,2)
	pred = net.predict(x_test)
	np.savetxt(outfile,pred,fmt="%i")

# if part == 'c':
