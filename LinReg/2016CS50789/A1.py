import numpy as np
from sklearn.linear_model import LassoLars
import sys
import pandas as pd


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

def MSELasso(Y,Yt):
	minval = np.amin(Y)
	err = np.sum(np.square(Yt - Y))
	denom = np.sum(np.square(Y-minval))
	return (err/denom)

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
	vals = [0.0001,0.001,0.01,0.1,0,1,10]
	errors = np.empty(7)
	for i in range(7):
		errors[i] = kFold(X,Y,vals[i])
	return vals[np.argmin(errors)]

part = sys.argv[1]
tr = sys.argv[2]
ts = sys.argv[3]
out = sys.argv[4]


if part == "a":
	train = np.loadtxt(open(tr,"rb"),delimiter=",")
	test = np.loadtxt(open(ts,"rb"),delimiter=",")

	x_train = np.insert(train[:,1:],0,1,axis=1)
	y_train = train[:,0:1]
	x_test = np.insert(test[:,1:],0,1,axis=1)
	y_test = test[:,0:1]

	W = trainCV(x_train,y_train,0)
	pred = np.matmul(x_test,W)
	# print (MSE(W,x_test,y_test))
	np.savetxt(out,pred)

if part == "b":
	train = np.loadtxt(open(tr,"rb"),delimiter=",")
	test = np.loadtxt(open(ts,"rb"),delimiter=",")
	
	x_train = np.insert(train[:,1:],0,1,axis=1)
	y_train = train[:,0:1]
	x_test = np.insert(test[:,1:],0,1,axis=1)
	y_test = test[:,0:1]

	lmbd = optLambda(x_train,y_train)
	# print (lmbd)
	W = trainCV(x_train,y_train,lmbd)
	pred = np.matmul(x_test,W)
	# print (MSE(W,x_test,y_test))
	np.savetxt(out,pred)

if part == "c":
	# train = np.load(tr)
	# test = np.load(ts)
	train = np.loadtxt(open(tr,"rb"),delimiter=",")
	test = np.loadtxt(open(ts,"rb"),delimiter=",")

	x_train = train[:,1:]
	y_train = train[:,0:1]
	x_test = test[:,1:]
	y_test = test[:,0:1]

	x_tr = np.concatenate((x_train,np.square(x_train),np.power(x_train,3)),axis=1)
	x_ts = np.concatenate((x_test,np.square(x_test),np.power(x_test,3)),axis=1)

	# print (MSELasso(y_test,pred.reshape((pred.size,1))))
	vals = [0.0000001,0.0001,1,10]
	errors = np.empty(4)

	for j in range (4):

		lm = vals[j]
		k=4
		err = np.empty(k)
		l = int(np.ma.size(x_train,axis=0)/k)
		x_cv,x_tr = np.split(x_train.copy(), [l], axis=0)
		y_cv,y_tr = np.split(y_train.copy(), [l], axis=0)
		model = LassoLars(alpha=lm)
		model.fit(x_tr,y_tr.ravel())
		pred = model.predict(x_cv)
		err[0] = MSELasso(y_cv,pred.reshape((pred.size,1)))

		for i in range(k-1):
			x_tr[i*l:(i+1)*l], x_cv = x_cv, x_tr[i*l:(i+1)*l].copy()
			y_tr[i*l:(i+1)*l], y_cv = y_cv, y_tr[i*l:(i+1)*l].copy()
			model = LassoLars(alpha=lm)
			model.fit(x_tr,y_tr.ravel())
			pred = model.predict(x_cv)
			err[i+1] = MSELasso(y_cv,pred.reshape((pred.size,1)))

		errors[j] = np.mean(err)

	x_tr = np.concatenate((x_train,np.square(x_train),np.power(x_train,3)),axis=1)
	optL = vals[np.argmin(errors)]
	# print (optL)
	model = LassoLars(alpha=optL)
	model.fit(x_tr,y_train.ravel())
	pred = model.predict(x_ts)
	# print (MSELasso(y_test,pred.reshape((pred.size,1))))
	np.savetxt(out,pred)
