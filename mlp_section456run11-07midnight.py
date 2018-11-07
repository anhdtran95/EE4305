import matplotlib.pyplot as plt


import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K

from keras.datasets import cifar10

#number of final classes (in cifar10, there are 10 classes)
nb_classes = 10
nb_epoch = 100
batch_size = 128

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.reshape(50000, 32 * 32 * 3)
X_test = X_test.reshape(10000, 32 * 32 * 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


def plot_train_acc(i, historyList):
    fig = plt.figure()
    for index, his in enumerate(historyList):
        plt.plot(range(nb_epoch),his.history['acc'],label='training'+str(index))
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('training_accuracy')
    plt.xlim([1,nb_epoch])
    plt.grid(True)
    plt.title("Training Accuracy Comparison")
    #plt.show()
    fig.savefig('img/'+str(i)+'-training-accuracy.png')
    plt.close(fig)
    
def plot_val_acc(i, historyList):
    fig = plt.figure()
    for index, his in enumerate(historyList):
        plt.plot(range(nb_epoch),his.history['val_acc'],label='validation'+str(index))
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('validation_accuracy')
    plt.xlim([1,nb_epoch])
    plt.grid(True)
    plt.title("Validation Accuracy Comparison")
    #plt.show()
    fig.savefig('img/'+str(i)+'-validation-accuracy.png')
    plt.close(fig)
    
def saveHistory(history, filename):
    import json
    json.dump(history.history, open('json_history/'+filename+'.json', 'w+'))

def MLP_learningRate(lr, decay):
    model = Sequential()
    model.add(Dense(512, input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=lr, momentum=0.0, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    #model.summary()
    return(model)

def MLP_act_func(act_func):
    model = Sequential()
    model.add(Dense(512, input_shape=X_train.shape[1:]))
    model.add(Activation(act_func))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation(act_func))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    #model.summary()
    return(model)

def MLP_loss_func(loss_func):
    model = Sequential()
    model.add(Dense(512, input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss=loss_func,
                  optimizer='sgd',
                  metrics=['accuracy'])
    #model.summary()
    return(model)

import json
historyDef = json.load(open('json_history/historyDef.json'))
# model 5 has 3 hiden layers
model5 = Sequential()
model5.add(Dense(1024, input_shape=X_train.shape[1:]))
model5.add(Activation('relu'))
model5.add(Dropout(0.2))
model5.add(Dense(512))
model5.add(Activation('relu'))
model5.add(Dropout(0.2))
model5.add(Dense(256))
model5.add(Activation('relu'))
model5.add(Dropout(0.2))
model5.add(Dense(10))
model5.add(Activation('softmax'))

model5.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history5 = model5.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test))
saveHistory(history5,'history5')
           
plot_train_acc(5 [historyDef, history5])
plot_val_acc(6, [historyDef, history5])

model6 = MLP_learningRate(0.3, 0.0)
model7 = MLP_learningRate(0.1, 0.1/nb_epoch)

history6 = model6.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test))
saveHistory(history6,'history6')
history7 = model7.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test))
saveHistory(history7,'history7')

plot_train_acc(7, [historyDef, history6, history7])
plot_val_acc(8, [historyDef, history6, history7])

model8 = MLP_act_func('sigmoid')
model9 = MLP_act_func('prelu')
model10 = MLP_act_func('softplus')

history8 = model8.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test))
saveHistory(history8,'history8')
history9 = model9.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test))
saveHistory(history9,'history9')
history10 = model10.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test))
saveHistory(history10,'history10')

plot_train_acc(7, [historyDef, history8, history9, history10])
plot_val_acc(8, [historyDef, history8, history9, history10])

model11 = MLP_loss_func('sparse_categorical_crossentropy')


history11 = model11.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_test, Y_test))
saveHistory(history11,'history11')

plot_train_acc(10, [historyDef, history11])
plot_val_acc(12, [historyDef, history11])