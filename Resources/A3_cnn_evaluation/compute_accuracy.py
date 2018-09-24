import sys
import numpy as np

def accuracy_score(gold, pred):
    num_samples = gold.size
    return np.sum(gold==pred)*100.0/num_samples

gold = np.loadtxt(sys.argv[1]).astype(np.int)
pred = np.loadtxt(sys.argv[2]).astype(np.int)

with open(sys.argv[3], 'w') as fp:
    acc = str(accuracy_score(gold, pred))
    print("Accuracy: %s" % acc)
    fp.write(acc)
