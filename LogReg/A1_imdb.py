import sys
import numpy as np 
import re
import math
from scipy.sparse import csr_matrix, dok_matrix, vstack
import scipy.sparse as sp
#-------------CONSTANTS----------------#

VOCAB = "Data/imdb_vocab"
TRAIN = "Data/imdb_train.csv"
TEST = "Data/imdb_test.csv"
#REGEX
re1 = re.compile("[^a-zA-Z0-9<>\s]+")
re2 = re.compile("<[^<>]+>")

#------------------FUNCTIONS------------------#
def extract_words(data):
	data = [re1.sub("", line.lower()) for line in data]
	data = [re2.sub(" ", line) for line in data]
	return data

def sigmoid(x):
	return 1/(1 + math.exp(-1*x))
sigmoid = np.vectorize(sigmoid)

def load_vocab(vocabulary):
	vocab = []
	for line in open(vocabulary, "r"):
		vocab.append(line.strip())
	dict = {j:i for i,j in enumerate(vocab)}
	return dict

def create_feat_mat(file,vocab):
	X = []
	Y = []
	for line in open(file, "r"):
		s = line.strip()
		Y.append(int(s[0:1]))
		X.append(s[1:])
	X = extract_words(X)	

	XF = dok_matrix((len(X),len(vocab.keys())+1),dtype=np.int8)
	length = len(X)
	for i in range(length):
		XF[i,0] = 1
		words = (X[i]).split()
		for word in words:
			if word in vocab:
				XF[i,vocab[word]+1] += 1
		# print (i)
	XF = XF.tocsr()
	return XF,(np.asarray(Y)).reshape((len(Y),1))

# def create_feat_mat2(file,vocab):
# 	X = []
# 	Y = []
# 	for line in open(file, "r"):
# 		s = line.strip()
# 		Y.append(int(s[0:1]))
# 		X.append(s[1:])
# 	X = extract_words(X)	

# 	XF = dok_matrix((len(X),len(vocab.keys())+1),dtype=np.int8)
# 	length = len(X)
# 	for i in range(length):
# 		XF[i,0] = 1
# 		words = (X[i]).split()
# 		for word in words:
# 			if word in vocab:
# 				XF[i,vocab[word]+1] += 1
# 		# print (i)
# 	XF = XF.tocsr()
# 	return XF,(np.asarray(Y)).reshape((len(Y),1))

def get_direction(X,W,Y):
	d = (X.transpose()).dot(Y - (X.dot(W)))
	norm = np.linalg.norm(d)
	return -1*d/norm

def rate_derivative(X,W,Y,rate,d):
	rt =  ((sigmoid(X.dot(W + rate*d)) - Y).transpose()).dot(X.dot(d))
	rtm = np.sum(rt)
	# print (rtm)
	return rtm

def exact_line_search(X,W,Y):
	r = 0.01
	eps = 0.001
	dirn = get_direction(X,W,Y)
	# print (dirn)
	while (rate_derivative(X,W,Y,r,dirn)) < 0:
		r = r*2
	r1 = 0
	r2 = r
	mid = (r1+r2)/2
	while np.absolute(r2-r1) > eps:
		if (rate_derivative(X,W,Y,mid,dirn)) > 0:
			r2 = mid
		else:
			r1 = mid
		mid = (r1+r2)/2
	return mid

def gradient_descent(X,Y,n,rate,lmbd,learn):
	#learn = 1 for constant, 2 for adaptive,using equation, 3 for binary
	W = np.zeros(((X.shape)[1],1))	
	# alpha = rate
	# ind = np.zeros((n,1))
	# log = np.zeros((n,1))
	for i in range(n):
		# ind[i] = i+1
		if(learn == 1):
			alpha = rate
		elif(learn == 2):
			alpha = rate/math.sqrt(i+1)
		else:
			alpha = exact_line_search(X,W,Y)

		grad_cost = (1/(X.shape)[0])*((X.transpose()).dot(sigmoid(X.dot(W))-Y))
		grad_reg = lmbd*W
		W = W - alpha*grad_cost
		W[1:] = W[1:] - alpha*grad_reg[1:]
		# log[i] = calc_cost(X,Y,W,lmbd)

	# plt.plot(ind,log)
	# plt.show()
	
	return W

def calc_cost(X,Y,W,lmbd):
	Ym = 1-Y
	YT = np.transpose(Y)
	YmT = np.transpose(Ym)
	XT = X.transpose()
	H = sigmoid(X.dot(W))
	S = ((YT.dot(np.log(H))) + (YmT.dot(np.log(1-H)))) + (lmbd/2)*(np.sum(np.square(W[1:])))
	return S

def findPredictions(X,W):
	return ((sigmoid(X.dot(W)))>0.5).astype(np.int)

def kFold(X,Y,n,rate,lmbd,learn,k=10):
	err = np.empty(k)
	l = int((X.shape[0])/k)

	for i in range(k):
		if i < k-1:
			x_cv = X[i*l:(i+1)*l]
			x_tr = vstack((X[:i*l],X[(i+1)*l:]))
			y_cv = Y[i*l:(i+1)*l]
			y_tr = vstack((Y[:i*l],Y[(i+1)*l:]))
		else:
			x_cv = X[i*l:]
			x_tr = X[:i*l]
			y_cv = Y[i*l:]
			y_tr = Y[:i*l]
		W = gradient_descent(x_tr,y_tr,n,rate,lmbd,learn)
		err[i] = calc_cost(x_cv,y_cv,W,lmbd)
 
	return np.mean(err)

def optimalReg(X,Y,n,rate,learn,k=10,lm=[0.00001,0.05,0.1,1,2]):
	errors = np.empty(len(lm))
	for i in range(len(lm)):
		errors[i] = -1*(kFold(X,Y,n,rate,lm[i],learn,k=k))
	return lm[np.argmin(errors)]

def stochastic(X,Y,n,rate,lmbd,learn,batch=128):
	l = int((X.shape[0])/batch)
	bt = 0
	W = np.zeros(((X.shape)[1],1))	

	for i in range(n):
		#select rate
		if(learn == 1):
			alpha = rate
		elif(learn == 2):
			alpha = rate/math.sqrt(i+1)
		else:
			alpha = exact_line_search(X,W,Y)

		#select batch
		if(bt < l-1):
			x_tr = X[bt*batch:(bt+1)*batch]
			y_tr = Y[bt*batch:(bt+1)*batch]
		else:
			x_tr = X[bt*batch:]
			y_tr = Y[bt*batch:]

		bt = (bt+1)%l

		#GD
		grad_cost = (1/(X.shape)[0])*((x_tr.transpose()).dot(sigmoid(x_tr.dot(W))-y_tr))
		grad_reg = lmbd*W
		W = W - alpha*grad_cost
		W[1:] = W[1:] - alpha*grad_reg[1:]

	return W

# def inplace_csr_row_normalize_l2(X):
#     """Inplace row normalize using the l2 norm"""
#     _inplace_csr_row_normalize_l2(X.data, X.shape, X.indices, X.indptr)


# def _inplace_csr_row_normalize_l2(X_data,shape,X_indices,X_indptr):
#     cdef integral n_samples = shape[0]
#     cdef integral n_features = shape[1]

#     cdef np.npy_intp i, j
#     cdef double sum_

#     for i in xrange(n_samples):
#         sum_ = 0.0

#         for j in xrange(X_indptr[i], X_indptr[i + 1]):
#             sum_ += (X_data[j] * X_data[j])

#         if sum_ == 0.0:
#             # do not normalize empty rows (can happen if CSR is not pruned
#             # correctly)
#             continue

#         sum_ = sqrt(sum_)

#         for j in xrange(X_indptr[i], X_indptr[i + 1]):
# X_data[j] /= sum_

def normalize(X):
	X = X.astype(float)    
	dat = X.data
	shape = X.shape
	indices = X.indices
	indptr = X.indptr

	n_samples = shape[0]
	n_features = shape[1]

	for i in range(n_samples):
		sum_ = 0.0
		for j in range(indptr[i],indptr[i+1]):
			sum_ += (dat[j]*dat[j])
		if(sum_ == 0.0):
			continue
		sum_ = np.sqrt(sum_)
		for j in range(indptr[i],indptr[i+1]):
			dat[j] /= sum_

	return csr_matrix((dat,indices,indptr),shape=(n_samples,n_features))


#------------------------CODE--------------------#
# vocabulary = load_vocab(VOCAB)
# x_train,y_train = create_feat_mat(TRAIN,vocabulary)
# x_test,y_test = create_feat_mat(TEST,vocabulary)

part = sys.argv[1]
method = int(sys.argv[2])
lr = float(sys.argv[3])
niter = int(sys.argv[4])
bs = int(sys.argv[5])
tr = sys.argv[6]
vocab = sys.argv[7]
ts = sys.argv[8]
out = sys.argv[9]

vocabulary = load_vocab(vocab)
x_train,y_train = create_feat_mat(tr,vocabulary)
x_train = normalize(x_train)
x_test,y_test = create_feat_mat(ts,vocabulary)
x_test = normalize(x_test)

if part == 'a':
	# lamb = optimalReg(x_train,y_train,100,lr,method)
	Wi = gradient_descent(x_train,y_train,niter,lr,0.00001,method)
	pred = (findPredictions(x_test,Wi)).astype(int)
	np.savetxt(out,pred,fmt="%i")

else:
	# lamb = optimalReg(x_train,y_train,100,lr,method)
	Wi = stochastic(x_train,y_train,niter,lr,0.00001,method,batch=bs)
	pred = (findPredictions(x_test,Wi)).astype(int)
	np.savetxt(out,pred,fmt="%i")