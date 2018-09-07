import numpy as np
import pandas as pd
import sys
import math
import re
from nltk.corpus import stopwords
# from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.stem.porter import PorterStemmer
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
import scipy as sp
# from sklearn.preprocessing import normalize
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.model_selection import train_test_split

# TRAIN_FILE = "../../../col341_a2_data/amazon_train.csv"
# TEST_FILE = "../../../col341_a2_data/amazon_test_public.csv"
RE1 = "\w+"
STOP = set(stopwords.words('english'))
BI_WORDS = set()
TRI_WORDS = set()
BI = False
TRI = False
# with open('stop2.txt',"r") as file:
#     STOP2=set(file.read().split())
# STOPLEV2 = False
# SNOW = SnowballStemmer("english")
SNOW = PorterStemmer()

def load_data(file_path,clip=False):
	dat = pd.read_csv(file_path,header=None,index_col=None)	
	dat = dat.replace(np.nan, '', regex=True)
	return {"reviews":dat[1].tolist(),"labels":pd.to_numeric(dat[0],downcast='integer').tolist()}

# def load_data_split(file_path,clip=False):
# 	dat = pd.read_csv(file_path,header=None,index_col=None)	
# 	dat = dat.replace(np.nan, '', regex=True)
# 	x_tr,x_ts,y_tr,y_ts = train_test_split(dat[1].tolist(),pd.to_numeric(dat[0],downcast='integer').tolist())
# 	return ({"reviews":x_tr,"labels":y_tr},{"reviews":x_ts,"labels":y_ts})

def filter_data(data,remove_stopwords,stemming):
	rev = data['reviews']
	filtered = []
	for line in rev:
		#Stop more
		words = [x for x in re.findall(RE1,line.lower())]
		# if (STOPLEV2):
		# 	words = list(set(words) - STOP2)
		# filtered_line = [SNOW.stem(w) for w in words if not w in STOP]
		if (remove_stopwords and stemming):
			filtered_line = [SNOW.stem(w) for w in words if not w in STOP]
		elif (remove_stopwords and not(stemming)):
			filtered_line = [w for w in words if not w in STOP]
		elif (not(remove_stopwords) and stemming):
			filtered_line = [SNOW.stem(w) for w in words]
		else:
			filtered_line = [w for w in words]
		filtered.append(" ".join(filtered_line))
	return {"reviews":filtered,"labels":data["labels"]}

# def trim_data(data,cc,minval):
# 	rev = data['reviews']
# 	lab = data['labels']
# 	trimmed = {'reviews':[],'labels':[]}
# 	for i in range(1,6):
# 		n = min(int(cc[i-1]),minval)
# 		tmp = [rev[j] for j in range(len(rev)) if lab[j] == i]
# 		trimmed['reviews'] += (np.random.choice(tmp,n)).tolist()
# 		trimmed['labels'] += [i for j in range(n)]
# 	return trimmed

def get_tokens(review,n_grams=1):
	tokens = [x for x in re.findall(RE1,review)]
	if(n_grams == 2):
		tokens = nltk.bigrams(tokens)
	if(n_grams == 3):
		tokens = nltk.trigrams(tokens)
	return tokens

def create_vocabulary(data,n_grams=1):
	vocab = set()
	revs = data['reviews']
	for review in revs:
		words = get_tokens(review,n_grams=n_grams)
		for word in words:
			vocab.add(word)

	return list(vocab)

def invert_vocab(vocab):
	return {vocab[i]:i for i in range(len(vocab))}

def train_model(data,vocab,inv_vocab):
	reviews = data["reviews"]
	labels = data["labels"]
	cc = np.zeros(len(set(labels)))
	wc = np.zeros((len(set(labels)),len(vocab)))
	ls = np.zeros(len(set(labels)))
	# wf = {vocab[i]:0 for i in range((len(vocab)))}
	for i in range(len(labels)):
		rev = reviews[i]
		lab = labels[i]
		cc[lab-1] += 1
		words = get_tokens(rev)
		biw = []
		tiw = []
		if (BI):
			biw = list(set(nltk.bigrams(words)) & BI_WORDS)
		if(TRI):
			tiw = list(set(nltk.bigrams(words)) & TRI_WORDS)
		words = words + biw + tiw
		for word in words:
			wc[lab-1][inv_vocab[word]] += 1
			ls[lab-1] += 1
	return {"cc":cc,"wc":wc,"ls":ls}

def train_model_tfidf(data,vocab,inv_vocab):
	reviews = data["reviews"]
	labels = data["labels"]
	cc = np.zeros(len(set(labels)))
	wc = np.zeros((len(set(labels)),len(vocab)))
	# ls = np.zeros(len(set(labels)))
	tf = lil_matrix((len(labels),len(vocab)))
	# wf = {vocab[i]:0 for i in range((len(vocab)))}
	for i in range(len(labels)):
		tmp = np.zeros(len(vocab))
		rev = reviews[i]
		lab = labels[i]
		cc[lab-1] += 1
		words = get_tokens(rev)
		biw = []
		tiw = []
		if (BI):
			biw = list(set(nltk.bigrams(words)) & BI_WORDS)
		if(TRI):
			tiw = list(set(nltk.bigrams(words)) & TRI_WORDS)
		words = words + biw + tiw
		for word in words:
			tmp[inv_vocab[word]] = 1
			tf[i,inv_vocab[word]] += 1
			# ls[lab-1] += 1
		wc[lab-1] += tmp
	# tf = (normalize(tf)).tocsr()
	tf = tf.tocsr()
	idf = np.log(len(labels) / (np.sum(wc,axis=0) + 1))
	# idf = 1/((np.sum(wc,axis=1))+1)
	tf = (tf.multiply(idf)).tocsr()
	# print (type(tf))

	# wc_n = [np.zeros(len(vocab)) for _ in range(len(set(labels)))]
	wc_n = np.zeros((len(set(labels)),len(vocab)))
	for i in range(len(labels)):
		wc_n[labels[i]-1] = wc_n[labels[i]-1] + tf[i]

	ls = np.sum(wc_n,axis=1)

	return {"cc":cc,"wc":wc_n,"ls":ls}

def choose_n_grams(data,vocab,inv_vocab,n,threshold=10):
	reviews = data["reviews"]
	wc = np.zeros(len(vocab))
	for i in range(len(reviews)):
		rev = reviews[i]
		words = get_tokens(rev,n_grams=n)
		for word in words:
			wc[inv_vocab[word]] += 1
	indices = (np.where(wc>threshold))[0]
	return [vocab[x] for x in indices]

def predict(review,model,vocab,inv_vocab):
	cc = model["cc"]
	tot = np.sum(cc)
	wc = model["wc"]
	ls = model["ls"]
	# wf = model["wf"]
	# norm = np.divide(wc,(np.sum(wc,axis=1)).reshape((5,1)))
	prob = np.zeros(len(cc))
	words = get_tokens(review)
	biw = []
	tiw = []
	if (BI):
		biw = list(set(nltk.bigrams(words)) & BI_WORDS)
	if(TRI):
		tiw = list(set(nltk.bigrams(words)) & TRI_WORDS)
	words = words + biw + tiw

	for label in range(len(cc)):
		tmp = np.log(cc[label]/tot)
		# tmp = 0
		den = ls[label] + len(vocab) + 1
		lc = wc[label]
		# den = cc[label] +  1
		for word in words:
			try:
				freq = lc[inv_vocab[word]]
				# occurence = wc[label][inv_vocab[word]]/wf[word]
			except:
				freq = 0
			tmp += np.log((freq+1)/den)
		prob[label] = tmp
	return np.argmax(prob)+1
##########################################################	

part = sys.argv[1]
tr = sys.argv[2]
ts = sys.argv[3]
outfile = sys.argv[4]

if part == 'a':
	train_data = filter_data(load_data(tr),False,False)
	test_data = filter_data(load_data(ts),False,False)
	vocab = create_vocabulary(train_data)
	inv_vocab = invert_vocab(vocab)
	model = train_model(train_data,vocab,inv_vocab)
	pred = [int(predict(rev,model,vocab,inv_vocab)) for rev in test_data["reviews"]]
	np.savetxt(outfile,pred,fmt="%i")

if part == 'b':
	# STOPLEV2 =True
	train_data = filter_data(load_data(tr),True,True)
	test_data = filter_data(load_data(ts),True,True)
	vocab = create_vocabulary(train_data)
	inv_vocab = invert_vocab(vocab)
	model = train_model(train_data,vocab,inv_vocab)
	pred = [int(predict(rev,model,vocab,inv_vocab)) for rev in test_data["reviews"]]
	np.savetxt(outfile,pred,fmt="%i")
	# STOPLEV2 = False

if part == 'c':
	#########TESTING############
	# STOPLEV2 = True
	# train_data = filter_data(load_data(tr),True,False)
	# test_data = filter_data(load_data(ts),True,False)
	# vocab = create_vocabulary(train_data['reviews'])
	# inv_vocab = invert_vocab(vocab)
	# model = train_model_tfidf(train_data,vocab,inv_vocab)
	# pred = [int(predict(rev,model,vocab,inv_vocab)) for rev in test_data["reviews"]]
	# np.savetxt(outfile,pred,fmt="%i")
	# STOPLEV2 = False
	##########TESTING############
	train_data = filter_data(load_data(tr),True,False)
	test_data = filter_data(load_data(ts),True,False)
	vocab_uni = create_vocabulary(train_data)
	vocab_bi = create_vocabulary(train_data,n_grams=2)
	vocab_bi_inv = invert_vocab(vocab_bi)
	vocab_tri = create_vocabulary(train_data,n_grams=3)
	vocab_tri_inv = invert_vocab(vocab_tri)
	vb = choose_n_grams(train_data,vocab_bi,vocab_bi_inv,2,threshold=20)
	vt = choose_n_grams(train_data,vocab_tri,vocab_tri_inv,3,threshold=10)
	# print (len(vb))
	# print (len(vt))
	# print ('bi words extracted')
	vocab = vocab_uni + vb + vt
	inv_vocab = invert_vocab(vocab)
	BI_WORDS = set(vb)
	BI = True
	TRI_WORDS = set(vt)
	TRI = True
	# print ("pre model step")
	model = train_model_tfidf(train_data,vocab,inv_vocab)
	pred = [int(predict(rev,model,vocab,inv_vocab)) for rev in test_data["reviews"]]
	np.savetxt(outfile,pred,fmt="%i")
	BI_WORDS = set()
	TRI_WORDS = set()
	BI = False
	TRI = False