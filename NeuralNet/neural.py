from scipy.special import expit, logit
import sys
import numpy as np
import pandas as pd
import math
import random
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier


train = pd.read_csv("../../col341_a2_data/devnagri_train.csv",header=None).values
test = pd.read_csv("../../col341_a2_data/devnagri_test_public.csv",header=None).values
# print (train.shape)
# print (test.shape)
x_train = train[:,1:]
x_test = test[:,1:]
y_train = train[:,:1]
y_test = test[:,:1]
one_hot = (LabelBinarizer()).fit(y_train)
y_tr = one_hot.transform(y_train)
y_ts = one_hot.transform(y_test)

net = MLPClassifier()