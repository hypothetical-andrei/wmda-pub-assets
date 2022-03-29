import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255

input_shape = (28, 28, 1)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv = Conv2D(28, 3, activation='relu')
    self.flatten = Flatten()
    self.dense1 = Dense(128, activation='relu')
    self.dense2 = Dense(10)
  def call(self, x):
    x = self.conv(x)
    x = self.flatten(x)
    x = self.dense1(x)
    return self.dense2(x)

model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)

EPOCHS = 10

for epoch in range(EPOCHS):
  print('STARTING epoch ', epoch)
  train_loss.reset_states()
  train_accuracy.reset_states()
  for images, labels in train_ds:
    train_step(images, labels)
  print('Loss: {loss}, accuracy {accuracy}'.format(loss=train_loss.result(), accuracy=train_accuracy.result()))