import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255

input_shape = (28, 28, 1)

model = Sequential([
  Conv2D(28, kernel_size=(3, 3), input_shape = input_shape),
  MaxPooling2D(pool_size = (2, 2)),
  Flatten(),
  Dense(128, activation=tf.nn.relu),
  Dropout(0.2),
  Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x = x_train, y = y_train, epochs=10)