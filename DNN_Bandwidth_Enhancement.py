#Loading the data
import scipy.io
x_train1l = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/training_rat_2/training_data2/01left/01left_bl.mat')
y_train1l = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/training_rat_2/training_data2/01left/01left_full.mat')
x_train1r = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/training_rat_2/training_data2/01right/01right_bl.mat')
y_train1r = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/training_rat_2/training_data2/01right/01right_full.mat')
x_train7l = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/training_rat_2/training_data2/07left/07left_bl.mat')
y_train7l = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/training_rat_2/training_data2/07left/07left_full.mat')
x_train7r = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/training_rat_2/training_data2/07right/07right_bl.mat')
y_train7r = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/training_rat_2/training_data2/07right/07right_full.mat')

import numpy 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Convolution1D
from keras.models import Model

seed = 7
numpy.random.seed(seed)

x_trainn1l = numpy.zeros((60700,512),dtype=[('keys','S3'),('values',float)])
x_trainn1l = numpy.array(x_train1l.values()[0])
y_trainn1l = numpy.zeros((60700,512),dtype=[('keys','S3'),('values',float)])
y_trainn1l = numpy.array(y_train1l.values()[0])

x_trainn1r = numpy.zeros((52800,512),dtype=[('keys','S3'),('values',float)])
x_trainn1r = numpy.array(x_train1r.values()[2])
y_trainn1r = numpy.zeros((52800,512),dtype=[('keys','S3'),('values',float)])
y_trainn1r = numpy.array(y_train1r.values()[0])

x_trainn7l = numpy.zeros((67300,512),dtype=[('keys','S3'),('values',float)])
x_trainn7l = numpy.array(x_train7l.values()[1])
y_trainn7l = numpy.zeros((67300,512),dtype=[('keys','S3'),('values',float)])
y_trainn7l = numpy.array(y_train7l.values()[0])

x_trainn7r = numpy.zeros((74300,512),dtype=[('keys','S3'),('values',float)])
x_trainn7r = numpy.array(x_train7r.values()[1])
y_trainn7r = numpy.zeros((74300,512),dtype=[('keys','S3'),('values',float)])
y_trainn7r = numpy.array(y_train7r.values()[2])

x_train = numpy.concatenate((x_trainn1l, x_trainn1r, x_trainn7l, x_trainn7r), axis=0)
y_train = numpy.concatenate((y_trainn1l, y_trainn1r, y_trainn7l, y_trainn7r), axis=0)

#Building the network
model = Sequential()
model.add(Dense(512, input_dim=512, init='normal', activation='relu'))
model.add(Dense(768, init='normal', activation='relu'))
model.add(Dense(256, init='normal', activation='relu'))
model.add(Dense(768, init='normal', activation='relu'))
model.add(Dense(512, init='normal'))

X_train = x_train.reshape(286300, 512)
Y_train = y_train.reshape(286300, 512)
model.compile(loss='mean_squared_error', optimizer='adam')

#training the model
history = model.fit(X_train, Y_train, batch_size=100, nb_epoch=100, verbose=1, callbacks=None, validation_split=0.1, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

#testing
x_test = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/testing_rat/bv_bl_1.mat')
y_test = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/testing_rat/bv_full_1.mat')
X_test = numpy.zeros((100,512),dtype=[('keys','S3'),('values',float)])
X_test = numpy.array(x_test.values()[1])
Y_test = numpy.zeros((100,512),dtype=[('keys','S3'),('values',float)])
Y_test = numpy.array(y_test.values()[0])
X_testr = X_test.reshape(100, 512)
Y_testr = Y_test.reshape(100, 512)
scores = model.evaluate(X_testr, Y_testr)
testPredict = model.predict(X_testr)
scipy.io.savemat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/predicted_rat/bv_pr_1.mat', {'testPredict':testPredict})

del x_test
del y_test
x_test = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/testing_rat/dz_bl_1.mat')
y_test = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/testing_rat/dz_full_1.mat')
X_test = numpy.zeros((100,512),dtype=[('keys','S3'),('values',float)])
X_test = numpy.array(x_test.values()[2])
Y_test = numpy.zeros((100,512),dtype=[('keys','S3'),('values',float)])
Y_test = numpy.array(y_test.values()[0])
X_testr = X_test.reshape(100, 512)
Y_testr = Y_test.reshape(100, 512)
scores = model.evaluate(X_testr, Y_testr)
testPredict = model.predict(X_testr)
scipy.io.savemat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/predicted_rat/dz_pr_1.mat', {'testPredict':testPredict})

del x_test
del y_test
x_test = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/testing_rat/pat_bl_1.mat')
y_test = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/testing_rat/pat_full_1.mat')
X_test = numpy.zeros((100,512),dtype=[('keys','S3'),('values',float)])
X_test = numpy.array(x_test.values()[2])
Y_test = numpy.zeros((100,512),dtype=[('keys','S3'),('values',float)])
Y_test = numpy.array(y_test.values()[0])
X_testr = X_test.reshape(100, 512)
Y_testr = Y_test.reshape(100, 512)
scores = model.evaluate(X_testr, Y_testr)
testPredict = model.predict(X_testr)
scipy.io.savemat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/predicted_rat/pat_pr_1.mat', {'testPredict':testPredict})


del x_test
del y_test
x_test = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/testing_rat/rat_bl_1.mat')
y_test = scipy.io.loadmat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/testing_rat/rat_full_1.mat')
X_test = numpy.zeros((100,512),dtype=[('keys','S3'),('values',float)])
X_test = numpy.array(x_test.values()[0])
Y_test = numpy.zeros((100,512),dtype=[('keys','S3'),('values',float)])
Y_test = numpy.array(y_test.values()[0])
X_testr = X_test.reshape(100, 512)
Y_testr = Y_test.reshape(100, 512)
scores = model.evaluate(X_testr, Y_testr)
testPredict = model.predict(X_testr)
scipy.io.savemat('/home/mig64gb1/Desktop/Neural_Networks/rat_brain/predicted_rat/rat_pr_1.mat', {'testPredict':testPredict})

#timing information
import time
start = time.time()
scores = model.evaluate(X_testr, Y_testr)
testPredict = model.predict(X_testr)
end = time.time()
print(end - start)
