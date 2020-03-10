import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

series = np.sin(0.1 * np.arange(200))
# plt.plot(series)
# plt.show()

T = 10
X = []
Y = []

for t in range(len(series) - T):
    x = series[t:t + T]
    X.append(x)
    y = series[t + T]
    Y.append(y)
X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)

i = tf.keras.layers.Input(shape=(T,))
x = tf.keras.layers.Dense(1)(i)
model = tf.keras.Model(i, x)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.1))
r = model.fit(X[:-N // 2], Y[:-N // 2],
              epochs=80,
              validation_data=(X[-N // 2:], Y[-N // 2:]))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()