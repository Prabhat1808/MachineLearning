import heapq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import math
import sys

def split_indices(data,idx):
    indices = []
    for val in range(np.max(data[:,idx]),np.min(data[:,idx])-1,-1):
        indices.append(np.where(data[:,idx]==val))
    return indices

def split_data(data,idx):
    splits = []
    n = 0
    for idxs in split_indices(data,idx):
        splits.append(data[idxs])
        n += (0 if len(data[idxs])<=0 else 1)
    if(n >= 2):
        return splits
    return None

def info_gain(data,attr_idx):
    splits = split_data(data,attr_idx)
    if splits==None:
        return -1
    info_gain = entropy(data)
    for i in range(len(splits)):
        if (len(splits[i])==0):
            continue
        prob = len(splits[i])/len(data)
        info_gain -= (prob*entropy(splits[i]))
    return info_gain

def entropy(data):
    num_pos = np.count_nonzero(data[:,0]==1)
    val_count = [num_pos,len(data)-num_pos]
    tot = sum(val_count)
    count = len(data)
    entropy = 0.0
    for val in val_count:
        if val > 0:
            tmp = val/tot
            entropy -= tmp*math.log(tmp,2)
    return entropy

def mispredicted(pred,lab):
    return (len(pred) - np.count_nonzero(pred==lab))

def checkLeafNode(data,feat):
    if (np.all(data[:,0]==data[0][0])) or (feat == None):
        return True
    return False

class Node:
    def __init__(self,train,height=0):
        self.train = train
        self.isLeaf = True
        self.split_info = {}
        self.split_info["split_feature"] = None
        self.split_info["split_feature_value"] = None # For numerical attr
        self.split_info["split_info_gain"] = 0
        self.height=height
        self.children = {}

        for i in range(1,self.train.shape[1]):
            split_val = None
            gain = info_gain(self.train,i)

            if(gain >= self.split_info["split_info_gain"]):
                self.split_info["split_info_gain"] = gain
                self.split_info["split_feature"] = i
                self.split_info["split_feature_value"] = split_val

    def grow_tree(self):

        if(checkLeafNode(self.train,self.split_info["split_feature"])):
            self.isLeaf = True
            return []

        self.isLeaf = False
        split_attr = self.split_info["split_feature"]
        splits = split_data(self.train,split_attr)

        if splits!=None:
            node_list = []
            for split in splits:
                if(len(split)==0):
                    continue
                new_node = Node(split,self.height+1)
                if (self.split_info["split_feature_value"] == None):
                    category = split[0][split_attr]
                else:
                    category = 0 if (split[0][split_attr]<=self.split_info["split_feature_value"]) else 1
                self.children[category] = new_node
                node_list.append(new_node)
            return node_list
        else:
            self.isLeaf=True
            return []

    def predict(self,data):
        pred = np.full(len(data),1 if ((self.train[:,0].mean())>0.5) else 0)
        self.misc_prune = mispredicted(pred,data[:,0])

        if(not self.isLeaf):
            for idx in split_indices(data,self.split_info["split_feature"]):
                if len(idx[0])==0:
                    continue
                split_val = data[idx][0][self.split_info["split_feature"]]
                if split_val in self.children:
                    preds = self.children[split_val].predict(data[idx])
                    pred[idx]=preds
                else:
                    continue

        self.misc_full = mispredicted(data[:,0],pred)
        self.gain_prune = self.misc_full - self.misc_prune

        return pred

    def get_subtree(self):
        if not(self.isLeaf):
            subtree = [self]
            for ch in self.children.keys():
                subtree += self.children[ch].get_subtree()
            return subtree
        else:
            return [self]

TR_DIR = "COL341_DecisionTree_data/train.csv"
VAL_DIR = "COL341_DecisionTree_data/valid.csv"
TS_DIR = "COL341_DecisionTree_data/test.csv"
DISCRETE = ['Work Class','Education','Marital Status','Occupation','Relationship','Race','Sex','Native Country']
CONTINUOUS = ['Age','Fnlwgt','Education Number','Capital Gain','Capital Loss','Hour per Week']


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

def computeThreshold(data, feature_name):
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

def load_data():
    tr = pd.read_csv(TR_DIR)
    val = pd.read_csv(VAL_DIR)
    ts = pd.read_csv(TS_DIR)
    tr_cols = tr.columns
    val_cols = val.columns
    ts_cols = ts.columns

    for i in range(len(tr_cols)):
        tr.rename(columns={tr_cols[i]:tr_cols[i].lstrip().rstrip()},inplace=True)

    for i in range(len(val_cols)):
        val.rename(columns={val_cols[i]:val_cols[i].lstrip().rstrip()},inplace=True)

    for i in range(len(ts_cols)):
        ts.rename(columns={ts_cols[i]:ts_cols[i].lstrip().rstrip()},inplace=True)

    for var in DISCRETE:
        tr[var] = tr[var].astype('category')
        tr[var] = tr[var].cat.codes
        val[var] = val[var].astype('category')
        val[var] = val[var].cat.codes
        ts[var] = ts[var].astype('category')
        ts[var] = ts[var].cat.codes

    thresholds = {}
    for col in CONTINUOUS:
        thresholds[col] = computeThreshold(tr,col)
        tr[col] = (tr[col] >= thresholds[col]).astype(int)
        val[col] = (val[col] >= thresholds[col]).astype(int)
        ts[col] = (ts[col] >= thresholds[col]).astype(int)

    dat_tr = np.array(tr,dtype=np.int64)
    dat_val = np.array(val,dtype=np.int64)
    dat_ts = np.array(ts,dtype=np.int64)

    return (dat_tr,dat_val,dat_ts)

def update_result(val,ind1,ind2,size,mispred):
    val[ind1][0] = size
    val[ind1][ind2] = mispred
    return val

def get_build_accuracy(root,tree,levels,level_size,merged_data):
    result = np.zeros((len(levels)+1,3),dtype=np.int32)
    for i in range(len(merged_data)):
        mispred = mispredicted(root.predict(merged_data[i]),merged_data[i][:,0])
        for lvl in range(len(levels)-1,-1,-1):
            result[lvl][0] = level_size[lvl]
            result[lvl][i+1] = mispred - sum([tree[idx].gain_prune for idx in levels[lvl]])
        result = update_result(result,-1,i+1,len(tree),mispred)
    return result

def prune(root,train,val,train_label,val_label):
    result = []
    while True:
        result.append([len(root.get_subtree()),mispredicted(root.predict(train),train_label),mispredicted(root.predict(val),val_label)])
        prune_node = None
        prune_gain = -1

        for node in root.get_subtree():
            if ((not(node.isLeaf)) and (node.gain_prune > prune_gain)):
                prune_node=node
                prune_gain = node.gain_prune
        if prune_node!=None:
            prune_node.isLeaf = True
        else:
            break

    return np.array(result)

def plot(misc,train,val,savefile,invert=False,title=None,path=None):
    divider = np.array([len(train),len(val)])
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
    plt.savefig(savefile)
    return (num_nodes,accuracy)

arguments = sys.argv
part = arguments[1]
TR_DIR = arguments[2]
VAL_DIR = arguments[3]
TS_DIR = arguments[4]
outfile = arguments[5]
plotfile = arguments[6]

train_dta, val_dta, test_dta = load_data()

trainlabels = train_dta[:,0]
vallabels = val_dta[:,0]

tree = []
leaves = []
levels = []
level_size = []

root = Node(train_dta)

tree.append(root)
heapq.heappush(leaves,(-root.split_info["split_info_gain"],0))

itr=0

while True:
    if(len(leaves)==0):
        break
    itr+=1
    neggain,idx = heapq.heappop(leaves)
    new_leaves = tree[idx].grow_tree()
    if(len(new_leaves)>0):
        levels.append([x[1] for x in leaves])
        level_size.append(len(tree))
    for l in tree[idx].grow_tree():
        newidx = len(tree)
        tree.append(l)
        heapq.heappush(leaves,(-l.split_info["split_info_gain"],newidx))

build_misc = get_build_accuracy(root,tree,levels,level_size,[train_dta,val_dta])

if (part == 'a'):
    title = "Nodes VS Accuracy [Without Pruning]"
    path = "../plots/grow_binary.pdf"

    np.savetxt(outfile,root.predict(test_dta),fmt="%i")
    build_numnodes,build_accuracy = plot(build_misc,train_dta,val_dta,plotfile,invert=False,title=title)
    # print("Max accuracy:\nTrain: {}\nVal: {}\n".format(*np.max(build_accuracy,axis=0)))

if(part == 'b'):
    prune_misc = prune(root,train_dta,val_dta,train_dta[:,0],val_dta[:,0])

    title = "Nodes VS Accuracy [With Pruning]"
    path = "../plots/prune_binary.pdf"

    np.savetxt(outfile,root.predict(test_dta),fmt="%i")
    # prune_numnodes,prune_accuracy = plot(prune_misc,train_dta,val_dta,test_dta,invert=True,title=title,path=path)
    prune_numnodes,prune_accuracy = plot(prune_misc,train_dta,val_dta,plotfile,invert=True,title=title)
    # print("Max accuracy:\nTrain: {}\nVal: {}\n".format(*np.max(prune_accuracy,axis=0)))
