import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('moore.csv', header=None).values
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1]
plt.scatter(X, Y)
plt.show()

Y = np.log(Y)
X = X - X.mean()

model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(1,)), tf.keras.layers.Dense(1)])
model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse')


# learning rate scheduler
def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001


scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
r = model.fit(X, Y, epochs=200, callbacks=[scheduler])

plt.plot(r.history['loss'], label='loss')
plt.show()

model.save('linearregression.h5')
model = tf.keras.models.load_model('linearregression.h5')
print(model.layers)
