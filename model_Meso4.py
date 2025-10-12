# model_Meso4.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dropout, Dense

def Meso4():
    model = Sequential()
    
    model.add(Conv2D(8, (3, 3), strides=(1, 1), padding='same', input_shape=(256, 256, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(8, (5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(16, (5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(16, (5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model
