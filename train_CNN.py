"""
related to CNN_detect 7,8
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from util import *



BATCH_SIZE = 100
NB_CLASSES = 2
NB_EPOCH = 5
img_rows, img_cols = (50, 50)
INPUT_SHAPE = (img_rows, img_cols, 1)
MODEL_SAVE_PATH = './CNN_model9.h5'

print("loading data...")
X, y = load_data(['./data/positive_data3.csv', './data/negative_not_empty4.csv'])
train_size = int(0.9 * X.shape[0])
X_train = X[0:train_size, :]
y_train = y[0:train_size]
X_test = X[train_size:, :]
y_test = y[train_size:]

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.transpose(0, 2, 3, 1)
X_test = X_test.transpose(0, 2, 3, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 65535.
X_test /= 65535.
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)


# define the CNN
model = Sequential()
model.add(Convolution2D(nb_filter=16, nb_row=3, nb_col=3, border_mode='valid', input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, shuffle=True,
          verbose=1, validation_split=0.1)

model.save(MODEL_SAVE_PATH)
"""
model = load_model(MODEL_SAVE_PATH)
"""
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
