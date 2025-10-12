# mesonet.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dropout, Dense

class Meso4:
    def __init__(self, learning_rate=0.001):
        self.model = Sequential([
            Conv2D(8, (3,3), strides=1, padding='same', input_shape=(256,256,3)),
            BatchNormalization(),
            Activation('relu'),
            AveragePooling2D(pool_size=(2,2), strides=2),

            Conv2D(8, (5,5), strides=1, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            AveragePooling2D(pool_size=(2,2), strides=2),

            Conv2D(16, (5,5), strides=1, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            AveragePooling2D(pool_size=(2,2), strides=2),

            Conv2D(16, (5,5), strides=1, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            AveragePooling2D(pool_size=(4,4), strides=4),

            Flatten(),
            Dropout(0.5),
            Dense(16),
            Activation('relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

    def load(self, weight_path):
        self.model.load_weights(weight_path)
        print(f"✅ Weights loaded from: {weight_path}")
