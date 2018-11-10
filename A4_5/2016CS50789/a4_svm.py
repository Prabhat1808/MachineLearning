import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from numpy import cov
import sys
import math

TRAIN = "COL341_SVM_data/DHC_train.csv"
TEST = "COL341_SVM_data/DHC_test.csv"

def LinearKernel(x1, x2,sigma=None):
    return np.dot(x1, x2)

def GaussianKernel(x, y, sigma=1.0):
    return (math.exp(sigma*((linalg.norm(x-y))**2)))

class SVM(object):
    def __init__(self,kernel,soft,gamma=0.01):
        self.kernel=kernel
        self.soft=soft
        self.gamma = -1*gamma

    def fit(self, X, y):
        self.X_tr = X
        self.y_tr = y
        self.n,self.m=X.shape
        Gram = np.array([[self.kernel(i,j,self.gamma) for j in X] for i in X])
        print (np.sum(Gram))

        P = cvxopt.matrix(np.outer(y,y) * Gram,(self.n,self.n),'d')
        q = cvxopt.matrix(np.ones(self.n) * -1)
        A = cvxopt.matrix(y, (1,self.n),'d')
        b = cvxopt.matrix(0.0)

        g1 = np.diag(np.ones(self.n) * -1)
        g2 = np.identity(self.n)
        G = cvxopt.matrix(np.vstack((g1, g2)))
        H1 = np.zeros(self.n)
        H2 = np.ones(self.n) * self.soft
        h = cvxopt.matrix(np.hstack((H1, H2)))

        model=cvxopt.solvers.qp(P, q, G, h, A, b)

        self.multipliers = np.ravel(model['x'])
        # print (self.multipliers)
        self.SV = np.array([x for x,y in enumerate(self.multipliers) if y > epslon])

        self.points = self.multipliers[self.SV]
        self.X = X[self.SV]
        self.Y = y[self.SV]

        self.weights=np.zeros(self.m)
        for i in range(len(self.points)):
            self.weights += self.points[i]*self.X[i]*self.Y[i]

        self.bias=0
        for i in range(len(self.points)):
            self.bias -= np.dot(self.weights,self.X[i]) - self.Y[i]
        self.bias= self.bias/self.SV.shape[0]

        return True

    def project(self,x):
        xn = np.array([np.sum((x-xtr)**2) for xtr in self.X_tr])
        return (np.dot((self.points*self.y_tr).transpose(),np.exp(self.gamma*xn)))

    def predict(self,data):
        if(self.kernel == LinearKernel):
            return np.sign(np.dot(data, self.weights) + self.bias)

        y_pred = []
        for i in data:
            y = self.project(i)
            # print (y)
            if y[0] > 0:
                y_pred.append(1)
            else:
                y_pred.append(-1)

        return np.asarray(y_pred)

# In[4]:


# print (X_tr[:,400])


# In[5]:


# ## BEST PARAMS ###
# SOFT = 0.01
# model = SVM(LinearKernel,SOFT)
# epslon = 0.00000001
# model.fit(X_tr,y_tr)
# y_pred = model.predict(X_ts)
# print ("Accuracy = ", float(np.sum(y_ts==y_pred))/y_ts.shape[0])
# ## BEST PARAMS ###


# In[6]:

#
# SOFT = 1
# gamma = 0.01
# epslon = 0.00001
# print ("building Model")
# model = SVM(GaussianKernel,SOFT,gamma=gamma)
# model.fit(X_tr,y_tr)
# print ("Done training")
# y_pred = model.predict(X_ts)
# print ("Accuracy = ", float(np.sum(y_ts==y_pred))/y_ts.shape[0])


# In[7]:




# In[8]:


# ## BEST PARAMS ###
# SOFT = 0.01
# model = SVM(LinearKernel,SOFT)
# epslon = 0.00000001
# model.fit(X_tr_pca,y_tr)
# y_pred = model.predict(X_ts_pca)
# print ("Accuracy = ", float(np.sum(y_ts==y_pred))/y_ts.shape[0])
## BEST PARAMS ###


# In[9]:

#
# SOFT = 1
# gamma = 0.01
# epslon = 0.00001
# print ("building Model")
# model = SVM(GaussianKernel,SOFT,gamma=gamma)
# model.fit(X_tr_pca,y_tr)
# print ("Done training")
# y_pred = model.predict(X_ts_pca)
# print ("Accuracy = ", float(np.sum(y_ts==y_pred))/y_ts.shape[0])

arguments = sys.argv
part = arguments[1]
TRAIN = arguments[2]
TEST = arguments[3]
outfile = arguments[4]
SOFT = float(arguments[5])
gamma = None
if part == 'b' or part == 'c':
    gamma = float(arguments[6])

epslon = 0.00000001
tr_dat = pd.read_csv(TRAIN,header=None).values
ts_dat = pd.read_csv(TEST,header=None).values
X_tr = tr_dat[:,1:]/255
X_ts = ts_dat[:,1:]/255
y_tr = np.sign(tr_dat[:,:1] - 0.5)

x_cov = cov(tr_dat[:,1:].T)
u, lmb, v = linalg.svd(x_cov)
X_tr_pca = np.dot(tr_dat[:,1:],u[:,:50])/255
X_ts_pca = np.dot(ts_dat[:,1:],u[:,:50])/255

if part == 'a':
    model = SVM(LinearKernel,SOFT)
    model.fit(X_tr,y_tr)
    y_pred = model.predict(X_ts)
    np.savetxt(outfile,(y_pred > 0).astype(int),fmt="%i")

if part == 'b':
    model = SVM(GaussianKernel,SOFT,gamma=gamma)
    model.fit(X_tr,y_tr)
    y_pred = model.predict(X_ts)
    np.savetxt(outfile,(y_pred > 0).astype(int),fmt="%i")

if part == 'c':
    model = SVM(GaussianKernel,SOFT,gamma=gamma)
    model.fit(X_tr_pca,y_tr)
    y_pred = model.predict(X_ts_pca)
    np.savetxt(outfile,(y_pred > 0).astype(int),fmt="%i")
