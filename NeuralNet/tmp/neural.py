import json
import random
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, normalize
from scipy.special import expit

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
      	return (a-y) * sigmoid_prime(z)

class neural_net(object):

    def __init__(self, sizes, cost=QuadraticCost):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in list(zip(self.sizes[:-1], self.sizes[1:]))]
        self.cost=cost

    def feedforward(self, a):
        for b, w in list(zip(self.biases, self.weights)):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print ("Epoch ",j," training complete")
            if monitor_training_cost and j%5 == 0:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print ("Cost on training data: ", cost)
            if monitor_training_accuracy and j%5 == 0:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print ("Accuracy on training data: ",accuracy,"/",n)
            if monitor_evaluation_cost and j%5 == 0:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print ("Cost on evaluation data:",cost)
            if monitor_evaluation_accuracy and j%5 == 0:
               	accuracy = self.accuracy(evaluation_data, convert=True)
                evaluation_accuracy.append(accuracy)
                print ("Accuracy on evaluation data: ",accuracy,"/",n_data)
            # print
        return evaluation_cost, evaluation_accuracy,training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in list(zip(nabla_b, delta_nabla_b))]
            nabla_w = [nw+dnw for nw, dnw in list(zip(nabla_w, delta_nabla_w))]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in list(zip(self.weights, nabla_w))]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in list(zip(self.biases, nabla_b))]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in list(zip(self.biases, self.weights)):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (self.cost).delta(zs[-1], activations[-1], y.reshape((46,1)))
        nabla_b[-1] = delta

        
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y)/len(data)
        return cost

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = neural_net(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return expit(z)

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

train = pd.read_csv("../dummy_evaluation/data/devnagri_train.csv",header=None).values
test = pd.read_csv("../dummy_evaluation/data/devnagri_test.csv",header=None).values
# print (train.shape)
# print (test.shape)
x_train = normalize(train[:,1:])
x_test = test[:,1:]
y_train = train[:,:1]
y_test = pd.read_csv('../dummy_evaluation/data/devnagri_target_labels.txt',header=None).values
one_hot = (LabelBinarizer()).fit(y_train)
y_tr = one_hot.transform(y_train)
y_ts = one_hot.transform(y_test)

# tr_tmp = [x.reshape((1024,1)) for x in x_train]
# ts_tmp = [x.reshape((1024,1)) for x in x_test]
# try_tmp = [y.reshape((46,1)) for y in y_tr]
# trs_tmp = [y.reshape((46,1)) for y in y_ts]

# train_data = list(zip(tr_tmp,try_tmp))
# test_data = list(zip(ts_tmp,trs_tmp))

# net = neural_net([1024,100,46])
# net.SGD(train_data,51,100,0.01,evaluation_data=test_data,monitor_evaluation_accuracy=True,monitor_evaluation_cost=True,monitor_training_accuracy=True,monitor_training_cost=True)
