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

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


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
    fig.savefig('img/'+str(i)+'-training-accuracy_cnn.png')
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
    fig.savefig('img/'+str(i)+'-validation-accuracy_cnn.png')
    plt.close(fig)
    
def saveHistory(history, filename):
    import json
    json.dump(history.history, open('json_history/'+filename+'.json', 'w+'))

modelDef = Sequential()
modelDef.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
modelDef.add(Activation('relu'))
modelDef.add(Conv2D(32,(3, 3)))
modelDef.add(Activation('relu'))
modelDef.add(MaxPooling2D(pool_size=(2, 2)))
modelDef.add(Dropout(0.25))

modelDef.add(Flatten())
modelDef.add(Dense(512))
modelDef.add(Activation('relu'))
modelDef.add(Dropout(0.5))
modelDef.add(Dense(nb_classes))
modelDef.add(Activation('softmax'))

modelDef.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

historyDef = modelDef.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(x_test, y_test))
saveHistory(historyDef,'historyDef_CNN')

model1 = Sequential()
model1.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model1.add(Activation('relu'))
model1.add(Conv2D(32,(3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(512))
#adding batch normalization before activation function
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(nb_classes))
model1.add(BatchNormalization())
#adding batch normalization before activation function
model1.add(Activation('softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history1 = model1.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(x_test, y_test))
saveHistory(history1,'history1_CNN')

plot_train_acc(1, [historyDef, history1])
plot_val_acc(2, [historyDef, history1])

def CNN_drop_rate(drop_rate1, drop_rate2):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate1))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate2))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

model3 = CNN_drop_rate(0.1,0.3)
model4 = CNN_drop_rate(0.5,0.5)

history3 = model3.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(x_test, y_test))
saveHistory(history3,'history3_CNN')
history4 = model4.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(x_test, y_test))
saveHistory(history4,'history4_CNN')

plot_train_acc(3, [historyDef, history3, history4])
plot_val_acc(4, [historyDef, history3, history4])

def CNN_learningRate(lr, decay):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=lr, momentum=0.0, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    #model.summary()
    return(model)

model6 = CNN_learningRate(0.3, 0.0)
model7 = CNN_learningRate(0.1, 0.1/nb_epoch)

history6 = model6.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(x_test, y_test))
saveHistory(history6,'history6')
history7 = model7.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(x_test, y_test))
saveHistory(history7,'history7')

plot_train_acc(7, [historyDef, history6, history7])
plot_val_acc(8, [historyDef, history6, history7])