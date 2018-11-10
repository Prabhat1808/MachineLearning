
# coding: utf-8

# In[1]:


import heapq
import numpy as np


class Node:
    def __init__(self,train_data,numerical_attr,depth=0):
        self.train_data = train_data
        self.isLeafNode = True
        self.numerical_attr=numerical_attr
        self.children = {}
        self.split_attr = None
        self.split_val = None # For numerical attr
        self.info_gain = 0
        self.majority = None
        self.depth=depth

        self.populate_majority()
        self.populate_split_attr()

    # Populate which of the two classes (0,1) is majority for this node
    def populate_majority(self):
        self.majority = 1 if ((self.train_data[:,0].mean())>0.5) else 0

    # Populate the attr which maximizes info_gain
    # sets the field info_gain,split_attr, split_attr_val
    def populate_split_attr(self):
        for i in range(1,self.train_data.shape[1]):
            if i in self.numerical_attr:
                split_val = self.calc_split_val(self.train_data,i)
            else:
                split_val = None
            gain = self.attr_info_gain(i,split_val)

            if(gain >= self.info_gain):
                self.info_gain = gain
                self.split_attr = i
                self.split_val = split_val

    # returns a list of np indices on which to split the data
    def split_idx(self,data,idx,split_val=None):
        if split_val != None:
            split_idx = [np.where(data[:,idx]<=split_val),np.where(data[:,idx]>split_val)]
        else:
            min_val = np.min(data[:,idx])
            max_val = np.max(data[:,idx])
            split_idx = []
            for val in range(min_val,max_val+1):
                split_idx.append(np.where(data[:,idx]==val))
        return split_idx

    # returns a list of data split on the idx given
    def split_data(self,data,idx,split_val=None):
        split_idx = self.split_idx(data,idx,split_val)
        splits = []
        num_nonzero = 0
        for idxs in split_idx:
            split = data[idxs]
            splits.append(split)
            num_nonzero += (1 if len(split)>0 else 0)
        if(num_nonzero < 2):
            # No splitting can be done on this index
            return None
        return splits

    # Calculate info gain w.r.t to an attribute for this node
    def attr_info_gain(self,attr_idx,split_val=None):
        splits = self.split_data(self.train_data,attr_idx,split_val)
        if splits==None:
            # No info gain as splitting does not result in multiple leaves
            return -1
        info_gain = self.calc_entropy(self.train_data)
        for i in range(len(splits)):
            if (len(splits[i])==0):
                continue
            prob = len(splits[i])/len(self.train_data)
            info_gain -= (prob*self.calc_entropy(splits[i]))
        return info_gain

    # calculates the median of the numericall attr for this node
    def calc_split_val(self,data,attr_idx):
        return np.median(data[:,attr_idx])

    # calculate number of misclassified examples
    def calc_misc(self,pred,label):
        return (len(pred) - np.count_nonzero(pred==label))
    # predict the label on this data
    def predict(self,data):
        pred = np.full(len(data),self.majority)

        # number of misclassified exampled if this node was pruned
        self.misc_prune = self.calc_misc(pred,data[:,0])

        if(not self.isLeafNode):
            # if not leaf node then recursively call children to calculate prediction
            split_idx = self.split_idx(data,self.split_attr,self.split_val)
            for idx in split_idx:
                if len(idx[0])==0:
                    continue
                split = data[idx]
                split_val = split[0][self.split_attr]
                if(self.split_attr in self.numerical_attr):
                    split_val = 1 if (split_val>self.split_val) else 0
                if split_val not in self.children:
    #                 # This split was not present in tranining data
    #                 print("WARNING: ")
    #                 print(split[0])
    #                 print("not recognizable by : ",self.split_attr)
    #                 print(self.children.keys())
    #                 print("---------------------")
                    continue
                preds = self.children[split_val].predict(split)
                pred[idx]=preds

        # number of misclassified exampled if this node fully grown
        self.misc_full = self.calc_misc(pred,data[:,0])
        self.gain_prune = self.misc_full - self.misc_prune

        return pred

    # Grows this node and returns an array of children
    # Should only be called after populate_split_attr
    def grow_tree(self):
        if (np.all(self.train_data[:,0]==self.train_data[0][0])):
            # all the labels are same and hence we should stop
#             print("exiting because all labels are same, depth: {}".format(self.depth))
            self.isLeafNode = True
            return []
        if((self.split_attr == None)):
#             print("exiting because no attribute to split on, depth: {}".format(self.depth))
            self.isLeafNode=True
            return []

        self.isLeafNode = False
        split_attr = self.split_attr
        splits = self.split_data(self.train_data,split_attr,self.split_val)
        if splits==None:
            self.isLeafNode=True
            return []
        node_list = []
        for split in splits:
            if(len(split)==0):
                continue
            new_node = Node(split,self.numerical_attr,self.depth+1)
            if (self.split_val != None):
                category = 1 if (split[0][split_attr]>self.split_val) else 0
            else:
                category = split[0][split_attr]
            self.children[category] = new_node
            node_list.append(new_node)
        return node_list

    # Calculates the entropy of the given sample
    # Assumes that the first column is the label
    def calc_entropy(self,data):
        num_pos = np.count_nonzero(data[:,0]==1)
        num_neg = len(data)-num_pos
        counts = [num_pos,num_neg]
        ent = 0
        for x in counts:
            if x > 0:
                prob = x/len(data)
                ent += (prob*np.log(prob))
        return -ent
    def get_subtree(self):
        if self.isLeafNode:
            return [self]
        subtree = [self]
        for ch in self.children.keys():
            subtree += self.children[ch].get_subtree()
        return subtree

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import pandas as pd
import numpy as np
import math
from sklearn import tree

"""
This file gives you the code to read the data into numpy arrays to get you startedf for part (a).
"""
from __future__ import print_function
import time,sys,statistics,csv
import numpy as np

## The possible attributes in the data with the prediction at index 0. Smaller names for brevity.
attributes = ["rich","age","wc","fnlwgt","edu","edun","mar","occ","rel","race","sex","capg","canpl","hpw","nc"]

## Get the encoding of the csv file by replacing each categorical attribute value by its index.
wc_l = "Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked".split(", ")
edu_l = "Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool".split(", ")
mar_l = "Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse".split(", ")
occ_l = "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces".split(", ")
rel_l = "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried".split(", ")
race_l = "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black".split(", ")
sex_l = "Female, Male".split(", ")
nc_l = "United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands".split(", ")
encode = {
    "rich"   : {"0":0,"1":1},
    "wc"     : {wc_l[i]:i for i in range(len(wc_l))},
    "edu"    : {edu_l[i]:i for i in range(len(edu_l))},
    "mar"    : {mar_l[i]:i for i in range(len(mar_l))},
    "occ"    : {occ_l[i]:i for i in range(len(occ_l))},
    "rel"    : {rel_l[i]:i for i in range(len(rel_l))},
    "race"   : {race_l[i]:i for i in range(len(race_l))},
    "sex"    : {sex_l[i]:i for i in range(len(sex_l))},
    "nc"     : {nc_l[i]:i for i in range(len(nc_l))},
    }

def medians(file):
    """
    Given a csv file, find the medians of the categorical attributes for the whole data.
    params(1):
        file : string : the name of the file
    outputs(6):
        median values for the categorical columns
    """
    fin = open(file,"r")
    reader = csv.reader(fin)
    age, fnlwgt, edun, capg, capl, hpw = ([] for i in range(6))
    total = 0
    for row in reader:
        total+=1
        if(total==1):
            continue
        l = [x.lstrip().rstrip() for x in row]
        age.append(int(l[0]));
        fnlwgt.append(int(l[2]));
        edun.append(int(l[4]));
        capg.append(int(l[10]));
        capl.append(int(l[11]));
        hpw.append(int(l[12]));
    fin.close()
    return(statistics.median(age),statistics.median(fnlwgt),statistics.median(edun),statistics.median(capg),statistics.median(capl),statistics.median(hpw))

def preprocess(file,binarize=True):
    """
    Given a file, read its data by encoding categorical attributes and binarising continuos attributes based on median.
    params(1):
        file : string : the name of the file
    outputs(6):
        2D numpy array with the data
    """
    # Calculate the medians
    agem,fnlwgtm,edunm,capgm,caplm,hpwm = medians(file)
    fin = open(file,"r")
    reader = csv.reader(fin)
    data = []
    total = 0
    for row in reader:
        total+=1
        # Skip line 0 in the file
        if(total==1):
            continue
        l = [x.lstrip().rstrip() for x in row]
        t = [0 for i in range(15)]

        # Encode the categorical attributes
        t[0] = encode["rich"][l[-1]]; t[2] = encode["wc"][l[1]]; t[4] = encode["edu"][l[3]]
        t[6] = encode["mar"][l[5]]; t[7] = encode["occ"][l[6]]; t[8] = encode["rel"][l[7]]
        t[9] = encode["race"][l[8]]; t[10] = encode["sex"][l[9]]; t[14] = encode["nc"][l[13]]

        # Binarize the numerical attributes based on median.
        # Modify this section to read the file in part c where you split the continuos attributes baed on dynamic median values.
        if binarize:
            t[1] = float(l[0])>agem; t[3] = float(l[2])>fnlwgtm; t[5] = float(l[4])>edunm;
            t[11] = float(l[10])>capgm; t[12] = float(l[11])>caplm; t[13] = float(l[12])>hpwm;
        else:
            t[1] = l[0]; t[3] = l[2]; t[5] = l[4];
            t[11] = l[10]; t[12] = l[11]; t[13] = l[12];

        # Convert some of the booleans to ints
        data.append([int(x) for x in t])

    return np.array(data,dtype=np.int64)

## Read the data
# train_data = preprocess("train.csv")
# valid_data = preprocess("valid.csv")
# test_data = preprocess("test.csv")

# print("The sizes are ","Train:",len(train_data),", Validation:",len(valid_data),", Test:",len(test_data))


# In[2]:


# get_ipython().magic('load_ext line_profiler')


# In[3]:


def num_mispred(pred,lab):
    return (len(pred) - np.count_nonzero(pred==lab))


# In[4]:


def get_build_accuracy(root,tree,levels,level_size,train,val):
    full_data = [train,val]test
    result = np.zeros((len(levels)+1,4),dtype=np.int32)
    for i in range(len(full_data)):
        data = full_data[i]
        label = data[:,0]
        pred = root.predict(data)
        mispred = num_mispred(pred,label)
        for lvl in range(len(levels)):
            result[lvl][0] = level_size[lvl]
            new_mispred = mispred - sum([tree[idx].gain_prune for idx in levels[lvl]])
            result[lvl][i+1] = new_mispred
        result[-1][0] = len(tree)
        result[-1][i+1] = mispred
    return result


# In[5]:


def prune(root,train,val):
    result = []
    train_label = train[:,0]
    val_label = val[:,0]
    # test_label = test[:,0]
    nodes = root.get_subtree()
    while True:
        train_misc = num_mispred(root.predict(train),train_label)
        # test_misc = num_mispred(root.predict(test),test_label)
        val_misc = num_mispred(root.predict(val),val_label) #important val should be last
        result.append([len(nodes),train_misc,val_misc])
        prune_node = None
        prune_gain = -1

        for node in nodes:
            if ((not(node.isLeafNode)) and (node.gain_prune > prune_gain)):
                prune_node=node
                prune_gain = node.gain_prune
        if prune_node==None:
            break
        prune_node.isLeafNode = True
        nodes = root.get_subtree()
    return np.array(result)


# In[6]:


def plot(misc,train,val,invert=False,title=None,path=None):
    divider = np.array([len(train),len(val),len(test)])
    accuracy =  (1 - misc[:,1:]/divider)
    num_nodes = misc[:,0]
    plt.plot(num_nodes,accuracy[:,0],label="train")
    plt.plot(num_nodes,accuracy[:,1],label="val")
    # plt.plot(num_nodes,accuracy[:,2],label="test")
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Accuracy")
    if title:
        plt.title(title)
    if invert:
        plt.gca().invert_xaxis()
    if path:
        plt.savefig(path,format="pdf",bbox_inches='tight')
    # plt.show()
    return (num_nodes,accuracy)


# In[7]:


TRAIN="../dataset/train.csv"
TEST="../dataset/test.csv"
VAL="../dataset/valid.csv"


# In[8]:


binarize=True
# binarize=True
train_dta = preprocess(TRAIN,binarize)
# test_dta = preprocess(TEST,binarize)
val_dta = preprocess(VAL,binarize)
# train_dta = train_dta[np.where(train_dta[:,0]==0)]
# testlabels = test_dta[:,0]
trainlabels = train_dta[:,0]
vallabels = val_dta[:,0]


# In[9]:


tree = []
leaves = []
levels = []
level_size = []


# In[10]:


if binarize:
    root = Node(train_dta,set())
else:
    root = Node(train_dta,set([1,3,5,11,12,13]))


# In[11]:


tree.append(root)
heapq.heappush(leaves,(-root.info_gain,0))


# In[12]:


iteration=0


# In[13]:


while len(leaves)!=0 :
    iteration+=1
    # indices of leaves at this iteration
    level = [x[1] for x in leaves]
    # size of tree at this level
    size = len(tree)
    neggain,idx = heapq.heappop(leaves)
    new_leaves = tree[idx].grow_tree()
    if(len(new_leaves)>0):
        levels.append(level)
        level_size.append(size)
    for l in new_leaves:
        newidx = len(tree)
        tree.append(l)
        heapq.heappush(leaves,(-l.info_gain,newidx))
#     print(len(tree))


# In[40]:


def get_multi_split(node):
    if node.isLeafNode:
        return {}
    result = {}
    for child in node.children:
        res = get_multi_split(node.children[child])
        for key in res.keys():
            prev_val = result.get(key,[])
            if(len(prev_val) < len(res[key])):
                result[key] = res[key]
    result[node.split_attr] = result.get(node.split_attr,[])+[node.split_val]
    return result


# In[43]:


max_numerical_splits = get_multi_split(root)


# In[44]:


max_numerical_splits


# In[45]:


build_misc = get_build_accuracy(root,tree,levels,level_size,train_dta,val_dta)


# In[46]:


if binarize:
    title = "Nodes VS Accuracy [Binary]"
    path = "../plots/grow_binary.pdf"
else:
    title = "Nodes VS Accuracy [Numerical]"
    path = "../plots/grow_numerical.pdf"
# build_numnodes,build_accuracy = plot(build_misc,train_dta,val_dta,test_dta,invert=False,title=title,path=path)
build_numnodes,build_accuracy = plot(build_misc,train_dta,val_dta,invert=False,title=title)
print("Max accuracy:\nTrain: {}\nVal: {}\nTest: {}\n".format(*np.max(build_accuracy,axis=0)))


# In[47]:


prune_misc = prune(root,train_dta,val_dta)


# In[48]:


if binarize:
    title = "Nodes VS Accuracy [Binary Pruning]"
    path = "../plots/prune_binary.pdf"
else:
    title = "Nodes VS Accuracy [Numerical Pruning]"
    path = "../plots/prune_numerical.pdf"
# prune_numnodes,prune_accuracy = plot(prune_misc,train_dta,val_dta,test_dta,invert=True,title=title,path=path)
prune_numnodes,prune_accuracy = plot(prune_misc,train_dta,val_dta,invert=True,title=title)
print("Max accuracy:\nTrain: {}\nVal: {}\n.format(*np.max(prune_accuracy,axis=0)))


# ### Scikit Decision tree

# In[89]:


def get_accuracy(max_depth,min_samples_split,min_samples_leaf,train,val):
    clf = DTC(criterion="entropy",max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
    clf.fit(train_dta[:,1:],train_dta[:,0])
    train_pred = clf.predict(train[:,1:])
    val_pred = clf.predict(val[:,1:])
    # test_pred = clf.predict(test[:,1:])
    train_acc = np.mean(train_pred==train[:,0])
    val_acc = np.mean(val_pred==val[:,0])
    # test_acc = np.mean(test_pred==test[:,0])
    print("max_depth: {} , min_samples_split: {}, min_samples_lead: {}".format(max_depth,min_samples_split,min_samples_leaf))
    print("Train Acc: {}\nVal Acc: {}\n".format(train_acc,val_acc))
    print("{},{},{},{},{}".format(max_depth,min_samples_split,min_samples_leaf,train_acc,val_acc))


# In[90]:


get_accuracy(1,	2,	1,train_dta,val_dta)
get_accuracy(2,	2,	1,train_dta,val_dta)
get_accuracy(5,	2,	1,train_dta,val_dta)
get_accuracy(7,	2,	1,train_dta,val_dta)
get_accuracy(10,	2,	1,train_dta,val_dta)
get_accuracy(15,	2,	1,train_dta,val_dta)
get_accuracy(10,	4,	1,train_dta,val_dta)
get_accuracy(10,	7,	1,train_dta,val_dta)
get_accuracy(10,	10,	1,train_dta,val_dta)
get_accuracy(10,	20,	1,train_dta,val_dta)
get_accuracy(10,	7,	2,train_dta,val_dta)
get_accuracy(10,	7,	4,train_dta,val_dta)
get_accuracy(10,	10,	4,train_dta,val_dta)
get_accuracy(10,	10,	10,train_dta,val_dta)
get_accuracy(10,	10,	15,train_dta,val_dta)
get_accuracy(10,	10,	20,train_dta,val_dta)


# In[110]:


get_accuracy(10,	4,	10,train_dta,val_dta)


# ## Scikit Random Forest

# In[115]:


def get_rfc_accuracy(n_estimators,max_features,bootstrap,train,val):
    clf = RFC(criterion="entropy",n_estimators=n_estimators,max_features=max_features,bootstrap=bootstrap)
    clf.fit(train_dta[:,1:],train_dta[:,0])
    train_pred = clf.predict(train[:,1:])
    val_pred = clf.predict(val[:,1:])
    # test_pred = clf.predict(test[:,1:])
    train_acc = np.mean(train_pred==train[:,0])
    val_acc = np.mean(val_pred==val[:,0])
    # test_acc = np.mean(test_pred==test[:,0])
    print("n_estimators: {} ,max_features: {} ,bootstrap: {}".format(n_estimators,max_features,bootstrap))
    print("Train Acc: {}\nVal Acc: {}\n".format(train_acc,val_acc))
    print("{},{},{},{},{}".format(n_estimators,max_features,bootstrap,train_acc,val_acc))


# In[141]:


get_rfc_accuracy(30,4,True,train_dta,val_dta)


# In[ ]:


# clf.fit(train_dta[:,1:],train_dta[:,0])


# In[ ]:


# pred = clf.predict(test_dta[:,1:])


# In[ ]:


# np.mean(pred==test_dta[:,0])


# In[ ]:
