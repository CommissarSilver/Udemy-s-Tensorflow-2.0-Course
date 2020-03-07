import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = tf.keras.datasets.mnist.load_data()
(X_train, Y_train), (X_test, Y_test) = data
X_train = X_train / 255
X_test = X_test / 255

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

print(model.evaluate(X_test, Y_test))
model.save('mnist_ann.h5')
