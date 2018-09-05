import numpy as np
import pandas as pd
import sys
import math
import re

TRAIN_FILE = "../../col341_a2_data/amazon_train.csv"
TEST_FILE = "../../col341_a2_data/amazon_test_public.csv"
RE1 = "\w+"

def load_data(file_path,clip=False):
	dat = pd.read_csv(file_path,header=None,index_col=None)	
	return {"reviews":dat[1].tolist(),"labels":pd.to_numeric(dat[0],downcast='integer').tolist()}

def get_tokens(review):
	try:
		# tokens = [x for x in review.split(' ')]
		tokens = [x for x in re.findall(RE1,review)]
	except:
		tokens = []
	return tokens

def create_vocabulary(data):
	vocab = set()
	for review in data:
		words = get_tokens(review)
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
	for i in range(len(labels)):
		tmp = np.zeros(len(vocab))
		rev = reviews[i]
		lab = labels[i]
		cc[lab-1] += 1
		words = get_tokens(rev)
		for word in words:
			tmp[inv_vocab[word]] = 1
		wc[lab-1] += tmp
	return {"cc":cc,"wc":wc}

def predict(review,model,vocab,inv_vocab):
	cc = model["cc"]
	tot = np.sum(cc)
	wc = model["wc"]
	prob = np.zeros(len(cc))
	words = get_tokens(review)
	for label in range(len(cc)):
		tmp = np.log(cc[label]/tot)
		# tmp = 0
		den = cc[label] + len(vocab) + 1
		# den = cc[label] +  1
		for word in words:
			try:
				occurence = wc[label][inv_vocab[word]]
			except:
				occurence = 0
			tmp += np.log((occurence+1)/den)
		prob[label] = tmp
	return np.argmax(prob)+1

train_data = load_data(TRAIN_FILE)
vocab = create_vocabulary(train_data["reviews"])
inv_vocab = invert_vocab(vocab)
model = train_model(train_data,vocab,inv_vocab)
test_data = load_data(TEST_FILE)
predictions = [int(predict(rev,model,vocab,inv_vocab)) for rev in test_data["reviews"]]