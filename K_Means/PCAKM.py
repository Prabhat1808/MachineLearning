import numpy as np
import sys
def get_matrix():
    trn=np.loadtxt(open("DHC_train.csv","rb"),delimiter=",",dtype="int64")
    return trn[:,0],trn[:,1:]
     
def get_PCA(mat, ax):
    src=mat-np.mean(mat,axis=0)
    u,s,v=np.linalg.svd(src)
    p_comp=v[:,:ax]
    print "PCA variance preserved:",(sum(s[:ax]/sum(s)))
    return(np.matmul(src,p_comp)),p_comp
    
def expect(points, centres, k):
    #this slows down the process but saves on memory requirement
    pairwise_distances=[]
    for i in range(k):
        all_distances=points-centres[i]
        d_sq=all_distances*all_distances
        d_sum=np.sum(d_sq,axis=1,dtype="int64")
        pairwise_distances.append(d_sum)
    pairwise_distances=np.array(pairwise_distances)
    print "Metric value:",sum(np.min(pairwise_distances,axis=0)**0.5)
    return np.argmin(pairwise_distances, axis=0)

def maximise(points, clusters, k):
    return np.array([points[clusters==i].mean(axis=0) for i in range(k)])

def init_random(points, k):
    #copy everything to stay safe
    init_arr=points.copy()
    np.random.shuffle(init_arr)
    return init_arr[-k:]

def get_entropy(vec,k):
    total=len(vec)*1.0
    probs=np.bincount(vec,minlength=k)/total
    return np.sum(np.nan_to_num(-probs*np.log(probs)))
   
def get_measures(res, gold, k):
    total=len(gold)*1.0
    pur=0
    cluster_entropy=get_entropy(res,k)
    data_entropy=get_entropy(gold,k)
    cond_entropy=data_entropy.copy()
    for i in range(k):
        cond_entropy-= len(gold[res==i])/total*get_entropy(gold[res==i],k)
        pur+=max(np.bincount(gold[res==i], minlength=k))
    print "Purity:",pur/total
    print "Entropy:",data_entropy-cond_entropy
    print "NMI:",2*cond_entropy/(data_entropy+cluster_entropy)
    
def run(p_val=200,k=2,steps=50, min_diff=0):
    train_y,x=get_matrix()
    points,p_comp=get_PCA(x,p_val)
    centres=init_random(points,k)
    clusters=np.array([0]*points.shape[0])
    for x in xrange(steps):
        print "Iteration ",x
        new_clusters=expect(points,centres,k)
        new_centres=maximise(points, new_clusters, k)
        #print np.linalg.norm(centres-new_centres,ord=0),"centres moved"
        diff=np.linalg.norm(clusters-new_clusters,ord=0)
        print diff,"points changed cluster"
        clusters=new_clusters
        get_measures(clusters,train_y,k)
        centres=new_centres
        if diff<=min_diff:
            break
    return clusters, centres, train_y, p_comp

def get_ground(clusters, gold, k=2):
    pur=[]
    for i in range(k):
        pur.append(np.argmax(np.bincount(gold[clusters==i], minlength=k)))
    return pur
if __name__=="__main__":
    np.warnings.filterwarnings('ignore')    
    clusters, centres, train_y, p_comp=run(int(sys.argv[1]))
    ground=get_ground(clusters,train_y)
    print "Cluster labels:",ground
    tst=np.loadtxt(open("DHC_test.csv",mode="rb"),delimiter=",",dtype="int64")
    tst_x=tst[:,1:]
    tst_x=tst_x-np.mean(tst_x,axis=0)
    tst_x=np.matmul(tst_x,p_comp)
    tst_clus=np.array([ground[x] for x in expect(tst_x,centres,2)])
    diff=np.linalg.norm(tst_clus-tst[:,0],ord=0)
    print "Test Accuracy:",1.0-(diff/len(tst[:,0]))    
    
