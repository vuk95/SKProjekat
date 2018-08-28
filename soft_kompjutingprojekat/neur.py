import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist
from projekat import crop_num
import cv2

def napravi_model(shape, n_classes):
    model = Sequential()
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=shape))
    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    return model


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #


    for i in range(len(X_test)):
        cropovana = crop_num(X_test[i])
        X_test[i] = cropovana
    for i in range(len(X_train)):
        cropovana = crop_num(X_train[i])
        X_train[i] = cropovana


    row, col = X_train.shape[1:]
    X_train = X_train.reshape(X_train.shape[0], row, col, 1)
    X_test = X_test.reshape(X_test.shape[0], row, col, 1)
    shape = (row, col, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # Scale the data to lie between 0 to 1
    X_train /= 255
    X_test /= 255

    n_classes = 10
    # iz int u kategoricki
    tr_lab_kat = to_categorical(y_train)
    te_lab_kat = to_categorical(y_test)

    model = napravi_model(shape, n_classes)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    hist = model.fit(X_train, tr_lab_kat, batch_size=256, epochs=30, verbose=1,
                             validation_data=(X_test, te_lab_kat))
    loss, gain = model.evaluate(X_test, te_lab_kat, verbose=0)
    print(gain)
    model.save_weights('model.h5')
