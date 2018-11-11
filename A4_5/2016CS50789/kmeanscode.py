import numpy as np
import sys
import pandas as pd
from numpy import cov

def get_cc(points,k):
    tmp = points.copy()
    clust = np.array([0]*tmp.shape[0])
    np.random.shuffle(tmp)
    return (tmp[-k:],clust)

def fit(points,train_y,k,steps, min_diff):

    centres,clusters=get_cc(points,k)
    for x in range(steps):
        new_clusters = np.argmin(np.array([(np.sum(((points-centres[i])**2),axis=1,dtype="int64")) for i in range(k)]), axis=0)
        new_centres = np.array([points[new_clusters==i].mean(axis=0) for i in range(k)])
        diff=np.linalg.norm(clusters-new_clusters,ord=0)
        clusters=new_clusters
        #############################
        total=(len(train_y)*1.0)
        pur=0
        cluster_entropy=np.sum(np.nan_to_num(-(np.bincount(clusters,minlength=k)/(len(clusters)*1.0))*np.log((np.bincount(clusters,minlength=k)/(len(clusters)*1.0)))))
        data_entropy=np.sum(np.nan_to_num(-(np.bincount(train_y,minlength=k)/(len(train_y)*1.0))*np.log((np.bincount(train_y,minlength=k)/(len(train_y)*1.0)))))
        cond_entropy=data_entropy.copy()
        for i in range(k):
            cond_entropy-= len(train_y[clusters==i])/(len(train_y)*1.0)*np.sum(np.nan_to_num(-(np.bincount(train_y[clusters==i],minlength=k)/(len(train_y[clusters==i])*1.0))*np.log((np.bincount(train_y[clusters==i],minlength=k)/(len(train_y[clusters==i])*1.0)))))
            pur+=max(np.bincount(train_y[clusters==i], minlength=k))
        ############################
        centres=new_centres
        if diff<min_diff:
            break
    return clusters, centres, train_y

arguments = sys.argv
part = arguments[1]
TRAIN_FILE = arguments[2]
TEST_FILE = arguments[3]
outfile = arguments[4]
if part != 'a':
    PC = int(arguments[5])

if part == 'a':
    train = np.load(TRAIN_FILE)
    test = np.load(TEST_FILE)
    points=(train[:,1:].astype("int32"))
    test_x=test
    k = 100
    iter = 20

else:
    train = pd.read_csv(TRAIN_FILE,header=None).values
    test = pd.read_csv(TEST_FILE,header=None).values
    xc = cov(train[:,1:].T)
    u, lmb, v = np.linalg.svd(xc)
    points = np.dot(train[:,1:],u[:,:PC])
    test_x = np.dot(test[:,1:],u[:,:PC])
    k = 100
    iter = 20

np.warnings.filterwarnings('ignore')
train_y=train[:,0]
clusters, centres, train_y=fit(points,train_y,k,iter,train.shape[0]*0.005)
ground = [(np.argmax(np.bincount(train_y[clusters==i], minlength=k))) for i in range(k)]
np.savetxt(outfile,np.asarray(np.array([ground[x] for x in np.argmin(np.array([(np.sum(((test_x-centres[i])**2),axis=1,dtype="int64")) for i in range(k)]), axis=0)])),fmt='%i')
