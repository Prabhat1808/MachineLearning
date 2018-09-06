import numpy as np
import pandas as pd
import sys
import math
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

TRAIN_FILE = "../../col341_a2_data/amazon_train.csv"
TEST_FILE = "../../col341_a2_data/amazon_test_public.csv"
RE1 = "\w+"
STOP = set(stopwords.words('english'))
with open('ranksnl_large.txt',"r") as fl:
    STOP2=set(fl.read().split())
STOPLEV2 = False
SNOW = SnowballStemmer("english")

def load_data(file_path,clip=False):
	dat = pd.read_csv(file_path,header=None,index_col=None)	
	dat = dat.replace(np.nan, '', regex=True)
	return {"reviews":dat[1].tolist(),"labels":pd.to_numeric(dat[0],downcast='integer').tolist()}

def filter_data(data,remove_stopwords,stemming):
	rev = data['reviews']
	filtered = []
	for line in rev:
		#Stop more
		words = [x for x in re.findall(RE1,line.lower())]
		if (STOPLEV2):
			words = list(set(words) - STOP2)
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

def trim_data(data,cc,minval):
	rev = data['reviews']
	lab = data['labels']
	trimmed = {'reviews':[],'labels':[]}
	for i in range(1,6):
		n = min(int(cc[i-1]),minval)
		tmp = [rev[j] for j in range(len(rev)) if lab[j] == i]
		trimmed['reviews'] += (np.random.choice(tmp,n)).tolist()
		trimmed['labels'] += [i for j in range(n)]
	return trimmed

def get_tokens(review,n_grams):
	tokens = [x for x in re.findall(RE1,review)]
	if(n_grams == 2):
		tokens = nltk.bigrams(tokens)
	if(n_grams == 3):
		tokens = nltk.trigrams(tokens)
	return tokens

def create_vocabulary(data,n_grams=1):
	vocab = set()
	for review in data:
		words = get_tokens(review,n_grams=n_grams)
		for word in words:
			vocab.add(word)

	return list(vocab)

def invert_vocab(vocab):
	return {vocab[i]:i for i in range(len(vocab))}

def train_model(data,vocab,inv_vocab,n_grams=1):
	reviews = data["reviews"]
	labels = data["labels"]
	cc = np.zeros(len(set(labels)))
	wc = np.zeros((len(set(labels)),len(vocab)))
	# wf = {vocab[i]:0 for i in range((len(vocab)))}
	for i in range(len(labels)):
		tmp = np.zeros(len(vocab))
		rev = reviews[i]
		lab = labels[i]
		cc[lab-1] += 1
		words = get_tokens(rev,n_grams=n_grams)
		for word in words:
			tmp[inv_vocab[word]] = 1
			# wf[word] += 1
		wc[lab-1] += tmp
	return {"cc":cc,"wc":wc}

def train_model_freq(data,vocab,inv_vocab,n_grams=1):
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
		words = get_tokens(rev,n_grams=n_grams)
		for word in words:
			wc[lab-1][inv_vocab[word]] += 1
			ls[lab-1] += 1
	return {"cc":cc,"wc":wc,"ls":ls}

def predict(review,model,vocab,inv_vocab,n_grams=1):
	cc = model["cc"]
	tot = np.sum(cc)
	wc = model["wc"]
	# wf = model["wf"]
	# norm = np.divide(wc,(np.sum(wc,axis=1)).reshape((5,1)))
	prob = np.zeros(len(cc))
	words = get_tokens(review,n_grams=n_grams)

	for label in range(len(cc)):
		tmp = np.log(cc[label]/tot)
		# tmp = 0
		den = cc[label] + len(vocab) + 1
		# den = cc[label] +  1
		for word in words:
			try:
				occurence = wc[label][inv_vocab[word]]
				# occurence = wc[label][inv_vocab[word]]/wf[word]
			except:
				occurence = 0
			tmp += np.log((occurence+1)/den)
		prob[label] = tmp
	return np.argmax(prob)+1

def predict_freq(review,model,vocab,inv_vocab,n_grams=1):
	cc = model["cc"]
	tot = np.sum(cc)
	wc = model["wc"]
	ls = model["ls"]
	# wf = model["wf"]
	# norm = np.divide(wc,(np.sum(wc,axis=1)).reshape((5,1)))
	prob = np.zeros(len(cc))
	words = get_tokens(review,n_grams=n_grams)

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

train_data = load_data(TRAIN_FILE)
test_data = load_data(TEST_FILE)
# vocab = create_vocabulary(train_n["reviews"])
# inv_vocab = invert_vocab(vocab)
# model = train_model(train_n,vocab,inv_vocab)
# train_n = removeStopwords(train_data)
# test_n = removeStopwords(test_data)

stopstem = [(False,False),(False,True),(True,False),(True,True)]
gram = [1,2,3]
stop2 = [True,False]
for i in range(4):
	stop = stopstem[i][0]
	stem = stopstem[i][1]
	print ("STOP :",stop,"\nSTEM :",stem,"\n---------")
	for k in range(2):
		STOPLEV2 = stop2[k]
		print ("STOP2 : ",STOPLEV2)
		train_n = filter_data(train_data,stop,stem)
		test_n = filter_data(test_data,stop,stem)
		for j in range(3):
			print ("GRAMS: ",j+1)
			vocab = create_vocabulary(train_n['reviews'],n_grams=j)
			inv_vocab = invert_vocab(vocab)
			model = train_model_freq(train_n,vocab,inv_vocab,n_grams=j)
			predictions = [int(predict_freq(rev,model,vocab,inv_vocab,n_grams=j)) for rev in test_n["reviews"]]
			print (np.bincount(predictions))
			np.savetxt("test_preds_freq/"+str(i)+""+str(k)+""+str(j)+".txt",predictions,fmt="%i")
# predictions = [int(predict(rev,model,vocab,inv_vocab)) for rev in test_n["reviews"]]