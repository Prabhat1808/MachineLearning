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