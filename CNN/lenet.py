from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers import Dropout
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import pandas as pd
import sys

class lenet:
	@staticmethod
	def build(rows, cols, numClasses,activation="relu"):
		model = Sequential()
		inputShape = (rows, cols, 1)

		if K.image_data_format() == "channels_first":
			inputShape = (1, rows, cols)

		model.add(Conv2D(16, 3, padding="same",	input_shape=inputShape,activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(32, 3, padding="same",activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(64, 3, padding="same",activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(500,activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(numClasses))
		model.add(Activation("softmax"))

		return model

arguments = sys.argv
tr = arguments[1]
ts = arguments[2]
outfile = arguments[3]

train = pd.read_csv(tr,header=None).values
test = pd.read_csv(ts,header=None).values

x_tr = train[:,1:].reshape(train.shape[0],32,32,1).astype('float32')/255
y_tr = np_utils.to_categorical(train[:,0])
x_ts = test[:,1:].reshape(test.shape[0],32,32,1).astype('float32')/255

opt = SGD(lr=0.01)
opt_adam = Adam()
model = lenet.build(32, 32, 46)
print("[INFO] compiling model...")
# model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=opt_adam,metrics=["accuracy"])

print("[INFO] training...")
model.fit(x_tr, y_tr, batch_size=128, epochs=100,verbose=1)

# print("[INFO] evaluating on training data...")
# (loss, accuracy) = model.evaluate(x_tr, y_tr, batch_size=128, verbose=1)
# print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

pred = model.predict_classes(x_ts)
np.savetxt(outfile,pred,fmt="%i")