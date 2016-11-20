def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

from scipy.misc import imread
import cPickle as pickle
import numpy as np
import os
from sklearn.cross_validation import train_test_split

def load_CIFAR_one(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        return X, np.array(Y, dtype=int)

def load_CIFAR10(PATH):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(PATH, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_one(f)
        xs.append(X)
        ys.append(Y)
    Xtemp = np.concatenate(xs)
    Ytemp = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_one(os.path.join(PATH, 'test_batch'))
    Xtr, Xv, Ytr, Yv = train_test_split(Xtemp, Ytemp, test_size=0.2, random_state=0)
    del Xtemp, Ytemp
    return Xtr, Ytr, Xte, Yte, Xv, Yv
#you need to add Xval
Xtr, Ytr, Xte, Yte, Xv, Yv = load_CIFAR10('.')
Xtr= Xtr[0:5000]
Ytr = Ytr[0:5000]


from sklearn.preprocessing import StandardScaler

def scaler_function(Xtr,Xt,Xv,scale=True):
    scaler = StandardScaler(with_std=scale).fit(Xtr)
    Xtr_scaled = scaler.transform(Xtr)
    Xt_scaled = scaler.transform(Xt)
    Xv_scaled = scaler.transform(Xv)
    return Xtr_scaled, Xt_scaled, Xv_scaled

Xtr, Xte, Xv = scaler_function(Xtr,Xte,Xv)

from keras.utils.np_utils import to_categorical

print Ytr
Ytr = to_categorical(Ytr)
Yte = to_categorical(Yte)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(100, input_dim=Xtr.shape[1], init='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, init='uniform', activation='softmax'))
model.compile(optimizer=SGD(lr=0.05), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(Xtr, Ytr, nb_epoch=50, batch_size=32, verbose=1, validation_data=(Xte,Yte))
scores = model.evaluate(Xte, Yte)
test_acc = scores[1]
print "hola"