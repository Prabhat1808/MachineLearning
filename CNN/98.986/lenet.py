from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers import Dropout
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import pandas as pd
import sys

class lenet:
	@staticmethod
	def build(numChannels, imgRows, imgCols, numClasses,activation="relu", weightsPath=None):
		# initialize the model
		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (numChannels, imgRows, imgCols)

		# define the first set of CONV => ACTIVATION => POOL layers
		model.add(Conv2D(16, 3, padding="same",	input_shape=inputShape))
		# model.add(Dropout(0.25))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(0.25))

		# define the second set of CONV => ACTIVATION => POOL layers
		######################################
		model.add(Conv2D(32, 3, padding="same"))
		# model.add(Dropout(0.25))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(32, 3, padding="same"))
		# model.add(Dropout(0.25))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# model.add(Conv2D(32, 3, padding="same"))
		# # model.add(Dropout(0.25))
		# model.add(Activation(activation))
		# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		#######################################
		model.add(Dropout(0.25))

		#######################################
		model.add(Conv2D(64, 3, padding="same"))
		# model.add(Dropout(0.25))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# model.add(Conv2D(64, 3, padding="same"))
		# # model.add(Dropout(0.25))
		# model.add(Activation(activation))
		# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		########################################
		model.add(Dropout(0.25))

		# define the first FC => ACTIVATION layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation(activation))
		model.add(Dropout(0.5))		
		# model.add(Dropout(0.25))

		# define the second FC layer
		model.add(Dense(numClasses))

		# lastly, define the soft-max classifier
		model.add(Activation("softmax"))

		# if a weights path is supplied (inicating that the model was
		# pre-trained), then load the weights
		if weightsPath is not None:
			model.load_weights(weightsPath)

		# return the constructed network architecture
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

#############OPTIMIZERS#################
opt = SGD(lr=0.01)
opt_adam = Adam()
opt_rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
opt_nadam = Nadam()
########################################
augment = ImageDataGenerator(rotation_range=10,zoom_range = 0.1,width_shift_range=0.1,height_shift_range=0.1)
augment.fit(x_tr)
###############CALLBACKS#################
annealer = ReduceLROnPlateau(patience=3, verbose=1, factor=0.5, min_lr=0.00001)
early_stopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='auto')
########################################


model = lenet.build(numChannels=1, imgRows=32, imgCols=32, numClasses=46,weightsPath=None)
print("[INFO] compiling model...")
# model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=opt_adam,metrics=["accuracy"])
# model.compile(loss="categorical_crossentropy", optimizer=opt_rms,metrics=["accuracy"])
# model.compile(loss="categorical_crossentropy", optimizer=opt_nadam,metrics=["accuracy"])

print("[INFO] training...")
# model.fit(x_tr, y_tr, batch_size=128, epochs=100,verbose=1, callbacks=[annealer])
model.fit_generator(augment.flow(x_tr, y_tr, batch_size=128), epochs=100,verbose=1, callbacks=[annealer])

# print("[INFO] evaluating on training data...")
# (loss, accuracy) = model.evaluate(x_tr, y_tr, batch_size=128, verbose=1)
# print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

pred = model.predict_classes(x_ts)
np.savetxt(outfile,pred,fmt="%i")