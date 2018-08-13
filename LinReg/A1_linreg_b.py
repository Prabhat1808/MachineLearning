import numpy as np
from sklearn.metrics import accuracy_score

train = np.load("col341_a1/msd_train.npy")
test = np.load("col341_a1/msd_test.npy")

x_train = np.insert(train[:,1:],0,1,axis=1)
y_train = train[:,0:1]
x_test = np.insert(test[:,1:],0,1,axis=1)
y_test = test[:,0:1]


def compareOutputs(y_te,y_pre):
	a = accuracy_score(y_te,y_pre)
	print("accuracy score : ",a)

def trainCV(X,Y,lmbd):
	n = np.ma.size(X,axis=0)
	m = np.ma.size(X,axis=1)
	X_T = np.matrix.transpose(X)
	Id = lmbd*np.identity(m)
	L1 = (1/n)*(np.matmul(X_T,X)) + Id
	inv = np.linalg.inv(L1)
	R1 = (1/n)*np.matmul(X_T,Y)
	return (np.matmul(inv,R1))

def MSE(W,X,Y):
	minval = np.amin(Y)
	err = np.sum(np.square(np.matmul(X,W) - Y))
	denom = np.sum(np.square(Y-minval))
	return (err/denom)

l = np.ma.size(x_train,axis=0)
x_cv,x_tr = np.split(x_train.copy(), [int(l/10)], axis=0)
y_cv,y_tr = np.split(y_train.copy(), [int(l/10)], axis=0)

def kFold(X,Y,lmbd,k=10):
	err = np.empty(k)
	l = int(np.ma.size(X,axis=0)/10)
	x_cv,x_tr = np.split(x_train.copy(), [l], axis=0)
	y_cv,y_tr = np.split(y_train.copy(), [l], axis=0)
	W = trainCV(x_tr,y_tr,lmbd)
	err[0] = MSE(W,x_cv,y_cv)

	for i in range(k-1):
		x_tr[i*l:(i+1)*l], x_cv = x_cv, x_tr[i*l:(i+1)*l].copy()
		y_tr[i*l:(i+1)*l], y_cv = y_cv, y_tr[i*l:(i+1)*l].copy()
		W = trainCV(x_tr,y_tr,lmbd)
		err[i+1] = MSE(W,x_cv,y_cv)

	return np.mean(err)

def optLambda(X,Y):
	vals = [0.01,0.01,0.1,0,1,10]
	for v in vals:
		print (v," : ",kFold(X,Y,v))