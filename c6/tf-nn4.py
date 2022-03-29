import tensorflow as tf
import os
from tensorflow.keras.layers import Layer

train_dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv';

train_dataset_raw = tf.keras.utils.get_file(fname = os.path.basename((train_dataset_url)), origin = train_dataset_url)

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(train_dataset_raw, batch_size, column_names=column_names, label_name=label_name, num_epochs=1)


def pack_features_vector(features, labels):
  features = tf.stack(list(features.values()), axis = 1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))
# print(features)

class MyCustomLayer(Layer):
  def __init__(self, units=128, initializer='glorot_uniform', activation=None, name=None, **kwargs):
    super(MyCustomLayer, self).__init__(**kwargs)
    self.units = units
    self.initializer = initializer
    self.activation = activation
    if name:
      self._name = name
  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.initializer, trainable=True)
    self.b = self.add_weight(shape=(self.units, ), initializer=self.initializer, trainable=True)
  def call(self, input_tensor):
    # result = tf.matmul(input_tensor, self.w) + self.b
    result = input_tensor @ self.w + self.b
    if (self.activation):
      result = self.activation(result)
    return result

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4, )),
  MyCustomLayer(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

predictions = model(features)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y, training):
  predicted = model(x, training=training)
  return loss_object(y_true=y, y_pred=predicted)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_value, grads = grad(model, features, labels)

optimizer.apply_gradients(zip(grads, model.trainable_variables))

EPOCHS=100

train_loss_results = []
train_accuracy_results = []

for epoch in range(EPOCHS):
  epoch_average_loss = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
  for (x, y) in train_dataset:
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    epoch_average_loss.update_state(loss_value)
    epoch_accuracy.update_state(y, model(x, training=True))
  print("Epoch {epoch} | Loss: | {loss} | Accuracy: {accuracy}".format(epoch=epoch, loss=epoch_average_loss.result(), accuracy=epoch_accuracy.result()))
  train_loss_results.append(epoch_average_loss.result())
  train_accuracy_results.append(epoch_accuracy.result())