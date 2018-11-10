import numpy as np

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

def run(k=46,steps=50, min_diff=0):
    train=np.load("train.npy")
    min_diff = train.shape[0]*0.0075
    points=train[:,1:].astype("int32")
    train_y=train[:,0]
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
        if diff<min_diff:
            break
    return clusters, centres, train_y

def get_ground(clusters, gold, k=46):
    pur=[]
    for i in range(k):
        pur.append(np.argmax(np.bincount(gold[clusters==i], minlength=k)))
    return pur
if __name__=="__main__":
    np.warnings.filterwarnings('ignore')
    clusters, centres, train_y=run()
    ground=get_ground(clusters,train_y)
    print "Cluster labels:",ground
    tst=np.load("test.npy")
    tst_x=tst[:,1:]
    tst_clus=np.array([ground[x] for x in expect(tst_x,centres,46)])
    diff=np.linalg.norm(tst_clus-tst[:,0],ord=0)
    print "Test Accuracy:",1.0-(diff/len(tst[:,0]))
