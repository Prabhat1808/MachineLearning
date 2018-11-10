import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt

TRAIN = "COL341_SVM_data/DHC_train.csv"
TEST = "COL341_SVM_data/DHC_test.csv"
TEST_LABELS = "COL341_SVM_data/orig_DHC_target_labels.txt"

def LinearKernel(x1, x2,sigma=None):
    return np.dot(x1, x2)

def GaussianKernel(x, y, sigma=1.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):
    def __init__(self,kernel,soft):
        self.kernel=kernel
        self.soft=soft

    def fit(self, TrainingInput, TrainingOutput):
        self.n,self.m=TrainingInput.shape
        Gram =np.array([[self.kernel(i,j) for j in TrainingInput] for i in TrainingInput])

        P = cvxopt.matrix(np.outer(TrainingOutput,TrainingOutput) * Gram,(self.n,self.n),'d')
        q = cvxopt.matrix(np.ones(self.n) * -1)
        A = cvxopt.matrix(TrainingOutput, (1,self.n),'d')
        b = cvxopt.matrix(0.0)

        g1 = np.diag(np.ones(self.n) * -1)
        g2 = np.identity(self.n)
        G = cvxopt.matrix(np.vstack((g1, g2)))
        H1 = np.zeros(self.n)
        H2 = np.ones(self.n) * self.soft
        h = cvxopt.matrix(np.hstack((H1, H2)))

        model=cvxopt.solvers.qp(P, q, G, h, A, b)

        self.multipliers = np.ravel(model['x'])
        self.SV = np.array([x for x,y in enumerate(self.multipliers) if y > epslon])

        self.points = self.multipliers[self.SV]
        self.X = TrainingInput[self.SV]
        self.Y = TrainingOutput[self.SV]

        self.weights=np.zeros(self.m)
        for i in range(len(self.points)):
            self.weights += self.points[i]*self.X[i]*self.Y[i]

        self.bias=0
        for i in range(len(self.points)):
            self.bias -= np.dot(self.weights,self.X[i]) - self.Y[i]
        self.bias= self.bias/self.SV.shape[0]

        return True

    def predict(self,data):
            return np.sign(np.dot(data, self.weights) + self.bias)


tr_dat = pd.read_csv(TRAIN,header=None).values
ts_dat = pd.read_csv(TEST,header=None).values
X_tr = tr_dat[:,1:]/255
X_ts = ts_dat[:,1:]/255
y_tr = np.sign(tr_dat[:,:1] - 0.5)
y_ts = np.sign(np.genfromtxt(TEST_LABELS) - 0.5)

SOFT = 0.001
model = SVM(LinearKernel,SOFT)
epslon = 0.0000001
model.fit(X_tr,y_tr)
y_pred = model.predict(X_ts)
print ("Accuracy = ", float(np.sum(y_ts==y_pred))/y_ts.shape[0])
