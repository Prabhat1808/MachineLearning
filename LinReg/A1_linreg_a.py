import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

train = np.load("col341_a1/msd_train.npy")
test = np.load("col341_a1/msd_test.npy")

y_train = train[:,0:1]
x_train = np.insert(train[:,1:],0,1,axis=1)
x_T = np.matrix.transpose(x_train)

y_test = test[:,0:1]
x_test = np.insert(test[:,1:],0,1,axis=1)

W = np.matmul((np.matmul((np.linalg.inv(np.matmul((x_T),x_train))),x_T)),y_train)

def compareOutputs(y_te,y_pre):
	a = accuracy_score(y_te,y_pre)
	print("accuracy score : ",a)

def MSE(W,X,Y):
	minval = np.amin(Y)
	err = np.sum(np.sqaure(np.matmul(X,W) - Y))
	denom = np.sum(np.sqaure(Y-minval))
	return (err/denom)

y_pre = np.around(np.matmul(x_test,W))
compareOutputs(y_pre,y_test)
	