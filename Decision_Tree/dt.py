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
    # print (val_count)
    # print ('YOOYOO')
    tot = sum(val_count)
    count = len(val_count)
    entropy = 0.0
    # print (tot)
    for val in val_count:
        tmp = (val/tot)
        entropy -= tmp*math.log(tmp,2)
    return entropy

def calcEntropyFeatures(column, threshold):
    n = column.shape[0]
    less = sum(column < threshold)
    more = n - less
    entropy = 0.0
    entropy -= (less/n)*math.log((less/n),2)
    entropy -= (more/n)*math.log((more/n),2)
    return entropy

def infoGain(data, feature_name, threshold):
    n = data.shape[0]
    less = data.loc[data[feature_name] < threshold]
    n_less = less.shape[0]
    more = data.loc[data[feature_name] >= threshold]
    n_more = more.shape[0]
    gain = calcEntropyLabels(data['Rich?'])
    if(n_less != 0):
        # print ("Caaling function less")
        # print (threshold)
        # print (n_less)
        gain -= (n_less/n)*calcEntropyLabels(less['Rich?'])
    if(n_more != 0):
        # print ("Caaling function more")
        # print (threshold)
        # print (n_more)
        gain -= (n_more/n)*calcEntropyLabels(more['Rich?'])
    return gain

def gainRatio(data, feature_name, threshold):
    return (infoGain(data,feature_name,threshold)/calcEntropyFeatures(data[feature_name],threshold))

def candidateThresholds(data, feature_name):
    sorted = (data[['Rich?',feature_name]].sort_values([feature_name], ascending=True)).values
    # df1 = df[['a','b']]
    # print ("YOOYOO")
    # print (sorted)
    labels = sorted[:,0]
    # print (labels)
    vals = sorted[:,1]
    # print (vals)
    th = set()
    for i in range(len(labels)-2):
        if(labels[i]!=labels[i+1]):
            # print ('Diff')
            th.add((vals[i]+vals[i+1])/2)
    return list(th)

def computeThreshold(data, feature_name)test_dta:
    candidates = candidateThresholds(data,feature_name)
    # print ("Candidates : ",candidates)
    best_th = 0.0
    max_gain = 0.0
    for th in candidates:
        i = infoGain(data, feature_name, th)
        if i > max_gain:
            max_gain = i
            best_th = th
    return best_th

def featToBool(data, feature_name):
    th = computeThreshold(data,feature_name)
    tmp = (data[feature_name] >= th).astype(int)
    return th

thresholds = {}
for col in CONTINUOUS:
    thresholds[col] = computeThreshold(tr,col)
    tr[col] = (tr[col] >= thresholds[col]).astype(int)
    val[col] = (val[col] >= thresholds[col]).astype(int)

dat_tr = np.array(tr,dtype=np.int64)
dat_val = np.array(val,dtype=np.int64)
