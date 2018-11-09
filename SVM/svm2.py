import numpy as np
import numpy
import pandas as pd
from numpy import linalg
import cvxopt
import cvxopt.solvers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

TRAIN_DIR = "COL341_SVM_data/DHC_train.csv"
TEST_DIR = "COL341_SVM_data/DHC_test.csv"

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


class SVM(object):
    def __init__(self,BoxConstant,kernel=linear_kernel):
        self.kernel=kernel
        self.BoxConstant=BoxConstant

    def fit(self, TrainingInput, TrainingOutput):
        SampleCount,self.FeatureCount=TrainingInput.shape
        GramMatrix=numpy.array([[self.kernel(i,j) for j in TrainingInput] for i in TrainingInput])

        P = cvxopt.matrix(numpy.outer(TrainingOutput,TrainingOutput) * GramMatrix,(SampleCount,SampleCount),'d')
        print ("P:",P.size)
        q = cvxopt.matrix(numpy.ones(SampleCount) * -1)
        print ("q:",q.size)
        A = cvxopt.matrix(TrainingOutput, (1,SampleCount),'d')
        print ("A:",A.size)
        b = cvxopt.matrix(0.0)
        print ("b:",b.size)
        G = cvxopt.matrix(numpy.vstack((numpy.diag(numpy.ones(SampleCount) * -1), numpy.identity(SampleCount))))
        print ("G:",G.size)
        h = cvxopt.matrix(numpy.hstack((numpy.zeros(SampleCount), numpy.ones(SampleCount) * self.BoxConstant)))
        print ("h:",h.size)
        Model=cvxopt.solvers.qp(P, q, G, h, A, b)
        print ("Solution :",Model)
        self.SupportVectorsIndices=numpy.array([x for x,y in enumerate(numpy.ravel(Model['x'])) if y>threshold])
        try:
            self.lamda=numpy.ravel(Model['x'])[self.SupportVectorsIndices]
            self.SupportVectors_X=TrainingInput[self.SupportVectorsIndices]
            self.SupportVectors_Y=TrainingOutput[self.SupportVectorsIndices]
            self.get_weights()
            self.get_bias()
            self.summary()
            return True
        except:
            return False

    def get_weights(self):
        self.weights=numpy.zeros(self.FeatureCount)
        for i,j,k in zip(self.lamda,self.SupportVectors_X,self.SupportVectors_Y):
            self.weights+=i*j*k

    def get_bias(self):
        self.bias=0
        for i,j,k in zip(self.lamda,self.SupportVectors_X,self.SupportVectors_Y):
            self.bias+=k-numpy.dot(self.weights,j)
        self.bias= self.bias/self.SupportVectorsIndices.shape[0]

    def summary(self):
        print ("Weights: ",self.weights)
        print ("Weights sum: ",numpy.sum(self.weights))
        print ("bias: ",self.bias)

    def predict(self,test):
            return numpy.sign(numpy.dot(test, self.weights) + self.bias)

threshold = 0.00001
train = pd.read_csv(TRAIN_DIR,header=None).values
test = pd.read_csv(TEST_DIR,header=None).values
X_tr = train[:,1:]
y_tr = train[:,:1]
X_ts = test[:,1:]
y_ts = test[:,:1]
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.33, random_state=42)
model = SVM(0.1)
model.fit(X_tr,y_tr)
y_pred = model.predict(X_tr)
print (accuracy_score(y_pred,y_tr))
