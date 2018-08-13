import pandas as pd
import numpy as np
import scipy as sp
import re
import os
import sys

dat = (pd.read_csv("Data/imdb_train.csv",header=None)).values

y_train = (dat[:,0:1]).ravel()
x_train = (dat[:,1:]).ravel()

with open("Data/imdb_vocab","r",encoding="utf8") as fl:
	vocab = fl.read().splitlines()
	fl.close
