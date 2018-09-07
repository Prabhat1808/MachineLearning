# vocab = create_vocabulary(train_n["reviews"])
# inv_vocab = invert_vocab(vocab)
# model = train_model(train_n,vocab,inv_vocab)
# train_n = removeStopwords(train_data)
# test_n = removeStopwords(test_data)
# train_n =filter_data(train_data,True,False)
# test_n = filter_data(test_data,True,False)
# #############################   PART A ######################################
# vocab = create_vocabulary(train_n['reviews'],n_grams=1)
# inv_vocab = invert_vocab(vocab)
# model = train_model_tfidf(train_n,vocab,inv_vocab,n_grams=1)
# predictions = [int(predict(rev,model,vocab,inv_vocab,n_grams=1)) for rev in train_n["reviews"]]
# print (np.bincount(predictions))
#################################Modified##################################33
# stopstem = [(False,False),(False,True),(True,False),(True,True)]
# gram = [1,2,3]
# stop2 = [True,False]
# for i in range(4):
# 	stop = stopstem[i][0]
# 	stem = stopstem[i][1]
# 	print ("STOP :",stop,"\nSTEM :",stem,"\n---------")
# 	for k in range(2):
# 		STOPLEV2 = stop2[k]
# 		print ("STOP2 : ",STOPLEV2)
# 		train_n = filter_data(train_data,stop,stem)
# 		test_n = filter_data(test_data,stop,stem)
# 		for j in range(3):
# 			print ("GRAMS: ",j+1)
# 			vocab = create_vocabulary(train_n['reviews'],n_grams=j)
# 			inv_vocab = invert_vocab(vocab)
# 			model = train_model_freq(train_n,vocab,inv_vocab,n_grams=j)
# 			predictions = [int(predict_freq(rev,model,vocab,inv_vocab,n_grams=j)) for rev in test_n["reviews"]]
# 			print (np.bincount(predictions))
# 			np.savetxt("test_preds_freq/"+str(i)+""+str(k)+""+str(j)+".txt",predictions,fmt="%i")
# predictions = [int(predict(rev,model,vocab,inv_vocab)) for rev in test_n["reviews"]]

# from sklearn.metrics import accuracy_score, f1_score

# train_n = filter_data(train_data,True,False)
# test_n = filter_data(test_data,True,False)
# vocab = create_vocabulary(train_n['reviews'],n_grams=2)
# inv_vocab = invert_vocab(vocab)
# model = train_model(train_n,vocab,inv_vocab,n_grams=2)
# predictions = [int(predict(rev,model,vocab,inv_vocab,n_grams=2)) for rev in test_n["reviews"]]
# print (np.bincount(predictions))
# np.savetxt("test_preds_freq/"+str(i)+""+str(k)+""+str(j)+".txt",predictions,fmt="%i")

# train_data,test_data = load_data_split(TRAIN_FILE)
# train_n =filter_data(train_data,True,False)
# test_n = filter_data(test_data,True,False)
# #############################   PART A ######################################
# vocab = create_vocabulary(train_n['reviews'],n_grams=1)
# inv_vocab = invert_vocab(vocab)
# model = train_model_tfidf(train_n,vocab,inv_vocab,n_grams=1)
# predictions = [int(predict(rev,model,vocab,inv_vocab,n_grams=1)) for rev in test_n["reviews"]]
# print (np.bincount(predictions))