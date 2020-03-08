import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# load dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train, X_test = X_train / 255, X_test / 255
print('The train dataset is of shape: {0}'.format(X_train.shape))
print('The test dataset is of shape: {0}'.format(X_test.shape))
print('Each instance is of shape: {0}'.format(X_test[0].shape))

# prepare the dataset
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
print('Now the train dataset is of shape: {0}'.format(X_train.shape))

# build the model
model = tf.keras.models.Sequential([tf.keras.layers.Input(X_train[0].shape),
                                    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=2, activation='relu'),
                                    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu'),
                                    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, activation='relu'),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10, activation='softmax')])

# compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=15)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

