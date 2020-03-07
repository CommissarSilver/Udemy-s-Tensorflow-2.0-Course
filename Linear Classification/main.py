import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

data = load_breast_cancer()  # load the breast cancer dataset from sklearn

print('We have {0} samples each with {1} features'.format(data.data.shape[0], data.data.shape[1]))

# the train_test_split function from sklearn allows us to well, split our dataset into train and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.2)

# using sklearn's standardscaler to scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# number of training examples and number of dimensions(features) for each data point
N, D = X_train.shape

# building the model and training it
model = tf.keras.Sequential([tf.keras.layers.Input(shape=(D,)),
                             tf.keras.layers.Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
r = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100)

print('Train Score: ', model.evaluate(X_train, Y_train))
print('Test score: ', model.evaluate(X_test, Y_test))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

p = model.predict(X_test)
p = np.round(p).flatten()

print('manually calculated accuracy: ', np.mean(p == Y_test))
print('evaluate output: ', model.evaluate(X_test, Y_test))

# saving model's weights
model.save('linearclassifier.h5')
