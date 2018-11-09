import pandas as pd
import numpy as np
import math
from sklearn import tree

model = tree.DecisionTreeClassifier()
TR_DIR = "COL341_DecisionTree_data/train.csv"
VAL_DIR = "COL341_DecisionTree_data/valid.csv"
TS_DIR = "COL341_DecisionTree_data/test.csv"
DISCRETE = ['Work Class','Education','Marital Status','Occupation','Relationship','Race','Sex','Native Country']
CONTINUOUS = ['Age','Fnlwgt','Education Number','Capital Gain','Capital Loss','Hour per Week']

tr = pd.read_csv(TR_DIR)
val = pd.read_csv(VAL_DIR)
tr_cols = tr.columns
val_cols = val.columns

for i in range(len(tr_cols)):
    tr.rename(columns={tr_cols[i]:tr_cols[i].lstrip().rstrip()},inplace=True)

for i in range(len(val_cols)):
    val.rename(columns={val_cols[i]:val_cols[i].lstrip().rstrip()},inplace=True)

for var in DISCRETE:
    tr[var] = tr[var].astype('category')
    tr[var] = tr[var].cat.codes
    val[var] = val[var].astype('category')
    val[var] = val[var].cat.codes

def calcEntropyLabels(labels):
    val_count = (labels.value_counts()).values
    tot = sum(val_count)
    count = len(val_count)
    entropy = 0.0
    for val in val_count:
        tmp = (val/tot)
        entropy -= tmp*log(tmp,count)
    return entropy

def calcEntropyFeatures(column, threshold):
    n = column.shape[0]
    less = sum(column < threshold)
    more = n - less
    entropy = 0.0
    entropy -= (less/n)*log((less/n),2)
    entropy -= (more/n)*log((more/n),2)
    return entropy

def infoGain(data, feature_name, threshold):
    n = data.shape[0]
    less = data.loc[data['feature_name'] < threshold]
    n_less = less.shape[0]
    more = data.loc[data['feature_name'] >= threshold]
    n_more = more.shape[0]
    gain = calcEntropy(data[' Rich?'])
    gain -= (n_less/n)*calcEntropy(less[' Rich?'])
    gain -= (n_more/n)*calcEntropy(more[' Rich?'])
    return gain

def gainRatio(data, feature_name, threshold):
    return (infoGain(data,feature_name,threshold)/calcEntropyFeatures(data[feature_name],threshold))

def candidateThresholds(data, feature_name):
    sorted = (data.sort_values([feature_name], ascending=True)).values
    labels = sorted[0]
    vals = sorted[1]
    th = set()
    for i in range(len(labels)-2):
        if(labels[i]!=labels[i+1]):
            th.add((vals[i]+vals[i+1])/2)
    return list(th)

def computeThreshold(data, feature_name):
    candidates = candidateThresholds(data,feature_name)
    best_th = 0.0
    max_gain = 0.0
    for th in candidates:
        i = infoGain(data, feature_name, th)
        if i > max_gain:
            max_gain = i
            best_th = th
    return th

def featToBool(data, feature_name):
    th = computeThreshold(data,feature_name)
    tmp = (data[feature_name] > th).astype(int)
    return tmp
