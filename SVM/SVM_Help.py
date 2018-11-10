
# coding: utf-8

# In[16]:


import numpy
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pandas
from functools import reduce
import matplotlib.pyplot as plt


# In[17]:


def LinearKernel(x1, x2,sigma=None): return numpy.dot(x1, x2)


# In[18]:


class SupportVectorMachine(object):
    def __init__(self,kernel,BoxConstant):
        self.kernel=kernel
        self.BoxConstant=BoxConstant

    def fit(self, TrainingInput, TrainingOutput):
        SampleCount,self.FeatureCount=TrainingInput.shape
        GramMatrix=numpy.array([[self.kernel(i,j) for j in TrainingInput] for i in TrainingInput])

        P = cvxopt.matrix(numpy.outer(TrainingOutput,TrainingOutput) * GramMatrix,(SampleCount,SampleCount),'d')

        q = cvxopt.matrix(numpy.ones(SampleCount) * -1)

        A = cvxopt.matrix(TrainingOutput, (1,SampleCount),'d')

        b = cvxopt.matrix(0.0)

        G = cvxopt.matrix(numpy.vstack((numpy.diag(numpy.ones(SampleCount) * -1), numpy.identity(SampleCount))))

        h = cvxopt.matrix(numpy.hstack((numpy.zeros(SampleCount), numpy.ones(SampleCount) * self.BoxConstant)))

        Model=cvxopt.solvers.qp(P, q, G, h, A, b)

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
        print "Weights: ",self.weights
        print "Weights sum: ",numpy.sum(self.weights)
        print "bias: ",self.bias

    def predict(self,test):
            return numpy.sign(numpy.dot(test, self.weights) + self.bias)


# In[19]:


TrainingOutput = numpy.sign(pandas.read_csv("SVM_Data/DHC_train.csv",header=None, usecols=[0]).values-0.5)
TrainingInput= pandas.read_csv("SVM_Data/DHC_train.csv",header=None, usecols=range(1,32*32+1)).values/255
TestingInput= pandas.read_csv("SVM_Data/DHC_test.csv",header=None, usecols=range(1,32*32+1)).values/255
ActualOutput=numpy.sign(numpy.genfromtxt("SVM_Data/target.txt")-0.5)


# In[20]:


#Cross Validation
Accuracy={}
for BoxConstant in [0.001,0.01,0.1,1,10,100]:
    DevnagariClassifier=SupportVectorMachine(LinearKernel,BoxConstant)
    #help(cvxopt.solvers.qp)
    threshold=0.0000001
    while(DevnagariClassifier.fit(TrainingInput,TrainingOutput)):
        DevnagariClassifier.summary()
        PredictedOutput=DevnagariClassifier.predict(TestingInput)
        Accuracy[BoxConstant,threshold]=float(numpy.sum(ActualOutput==PredictedOutput))/ActualOutput.shape[0]
        print "BoxConstant: ",BoxConstant," threshold: ",threshold," Accuracy: ",Accuracy[BoxConstant,threshold]
        threshold=threshold*10


