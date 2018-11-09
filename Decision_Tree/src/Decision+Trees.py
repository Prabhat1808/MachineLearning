
# coding: utf-8

# In[1]:


from read_data import *
from node import Node
import heapq
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np


# In[2]:


# get_ipython().magic('load_ext line_profiler')


# In[3]:


def num_mispred(pred,lab):
    return (len(pred) - np.count_nonzero(pred==lab))


# In[4]:


def get_build_accuracy(root,tree,levels,level_size,train,val,test):
    full_data = [train,val,test]
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


def prune(root,train,val,test):
    result = []
    train_label = train[:,0]
    val_label = val[:,0]
    test_label = test[:,0]
    nodes = root.get_subtree()
    while True:
        train_misc = num_mispred(root.predict(train),train_label)
        test_misc = num_mispred(root.predict(test),test_label)
        val_misc = num_mispred(root.predict(val),val_label) #important val should be last
        result.append([len(nodes),train_misc,val_misc,test_misc])
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


def plot(misc,train,val,test,invert=False,title=None,path=None):
    divider = np.array([len(train),len(val),len(test)])
    accuracy =  (1 - misc[:,1:]/divider)
    num_nodes = misc[:,0]
    plt.plot(num_nodes,accuracy[:,0],label="train")
    plt.plot(num_nodes,accuracy[:,1],label="val")
    plt.plot(num_nodes,accuracy[:,2],label="test")
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Accuracy")
    if title:
        plt.title(title)
    if invert:
        plt.gca().invert_xaxis()
    if path:
        plt.savefig(path,format="pdf",bbox_inches='tight')
    plt.show()
    return (num_nodes,accuracy)


# In[7]:


TRAIN="../dataset/train.csv"
TEST="../dataset/test.csv"
VAL="../dataset/valid.csv"


# In[8]:


binarize=False
# binarize=True
train_dta = preprocess(TRAIN,binarize)
test_dta = preprocess(TEST,binarize)
val_dta = preprocess(VAL,binarize)
# train_dta = train_dta[np.where(train_dta[:,0]==0)]
testlabels = test_dta[:,0]
trainlabels = train_dta[:,0]


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


build_misc = get_build_accuracy(root,tree,levels,level_size,train_dta,val_dta,test_dta)


# In[46]:


if binarize:
    title = "Nodes VS Accuracy [Binary]"
    path = "../plots/grow_binary.pdf"
else:
    title = "Nodes VS Accuracy [Numerical]"
    path = "../plots/grow_numerical.pdf"
# build_numnodes,build_accuracy = plot(build_misc,train_dta,val_dta,test_dta,invert=False,title=title,path=path)
build_numnodes,build_accuracy = plot(build_misc,train_dta,val_dta,test_dta,invert=False,title=title)
print("Max accuracy:\nTrain: {}\nVal: {}\nTest: {}\n".format(*np.max(build_accuracy,axis=0)))


# In[47]:


prune_misc = prune(root,train_dta,val_dta,test_dta)


# In[48]:


if binarize:
    title = "Nodes VS Accuracy [Binary Pruning]"
    path = "../plots/prune_binary.pdf"
else:
    title = "Nodes VS Accuracy [Numerical Pruning]"
    path = "../plots/prune_numerical.pdf"
# prune_numnodes,prune_accuracy = plot(prune_misc,train_dta,val_dta,test_dta,invert=True,title=title,path=path)
prune_numnodes,prune_accuracy = plot(prune_misc,train_dta,val_dta,test_dta,invert=True,title=title)
print("Max accuracy:\nTrain: {}\nVal: {}\nTest: {}\n".format(*np.max(prune_accuracy,axis=0)))


# ### Scikit Decision tree

# In[89]:


def get_accuracy(max_depth,min_samples_split,min_samples_leaf,train,val,test):
    clf = DTC(criterion="entropy",max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
    clf.fit(train_dta[:,1:],train_dta[:,0])
    train_pred = clf.predict(train[:,1:])
    val_pred = clf.predict(val[:,1:])
    test_pred = clf.predict(test[:,1:])
    train_acc = np.mean(train_pred==train[:,0])
    val_acc = np.mean(val_pred==val[:,0])
    test_acc = np.mean(test_pred==test[:,0])
    print("max_depth: {} , min_samples_split: {}, min_samples_lead: {}".format(max_depth,min_samples_split,min_samples_leaf))
    print("Train Acc: {}\nVal Acc: {}\nTest Acc: {}".format(train_acc,val_acc,test_acc))
    print("{},{},{},{},{},{}".format(max_depth,min_samples_split,min_samples_leaf,train_acc,val_acc,test_acc))


# In[90]:


get_accuracy(1,	2,	1,train_dta,val_dta,test_dta)
get_accuracy(2,	2,	1,train_dta,val_dta,test_dta)
get_accuracy(5,	2,	1,train_dta,val_dta,test_dta)
get_accuracy(7,	2,	1,train_dta,val_dta,test_dta)
get_accuracy(10,	2,	1,train_dta,val_dta,test_dta)
get_accuracy(15,	2,	1,train_dta,val_dta,test_dta)
get_accuracy(10,	4,	1,train_dta,val_dta,test_dta)
get_accuracy(10,	7,	1,train_dta,val_dta,test_dta)
get_accuracy(10,	10,	1,train_dta,val_dta,test_dta)
get_accuracy(10,	20,	1,train_dta,val_dta,test_dta)
get_accuracy(10,	7,	2,train_dta,val_dta,test_dta)
get_accuracy(10,	7,	4,train_dta,val_dta,test_dta)
get_accuracy(10,	10,	4,train_dta,val_dta,test_dta)
get_accuracy(10,	10,	10,train_dta,val_dta,test_dta)
get_accuracy(10,	10,	15,train_dta,val_dta,test_dta)
get_accuracy(10,	10,	20,train_dta,val_dta,test_dta)


# In[110]:


get_accuracy(10,	4,	10,train_dta,val_dta,test_dta)


# ## Scikit Random Forest

# In[115]:


def get_rfc_accuracy(n_estimators,max_features,bootstrap,train,val,test):
    clf = RFC(criterion="entropy",n_estimators=n_estimators,max_features=max_features,bootstrap=bootstrap)
    clf.fit(train_dta[:,1:],train_dta[:,0])
    train_pred = clf.predict(train[:,1:])
    val_pred = clf.predict(val[:,1:])
    test_pred = clf.predict(test[:,1:])
    train_acc = np.mean(train_pred==train[:,0])
    val_acc = np.mean(val_pred==val[:,0])
    test_acc = np.mean(test_pred==test[:,0])
    print("n_estimators: {} ,max_features: {} ,bootstrap: {}".format(n_estimators,max_features,bootstrap))
    print("Train Acc: {}\nVal Acc: {}\nTest Acc: {}".format(train_acc,val_acc,test_acc))
    print("{},{},{},{},{},{}".format(n_estimators,max_features,bootstrap,train_acc,val_acc,test_acc))


# In[141]:


get_rfc_accuracy(30,4,True,train_dta,val_dta,test_dta)


# In[ ]:


clf.fit(train_dta[:,1:],train_dta[:,0])


# In[ ]:


pred = clf.predict(test_dta[:,1:])


# In[ ]:


np.mean(pred==test_dta[:,0])


# In[ ]:
