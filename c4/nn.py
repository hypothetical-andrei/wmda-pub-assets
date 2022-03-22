import numpy as np
import random

class Network:
  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
  def feedforward(self, a):
    for b, w in zip(self.biases, self.weights):
      a = sigmoid(np.dot(w, a) + b)
    return a
  def backprop(self, x, y):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation) + b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    for l in range(2, self.num_layers):
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)
  def update_mini_batch(self, mini_batch, eta):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
      delta_b, delta_w = self.backprop(x, y)
      nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
      nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]
    self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
  def gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data = None):
    training_data = list(training_data)
    n = len(training_data)
    if test_data:
      test_data = list(test_data)
      n_test = len(test_data)
    for i in range(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
      if test_data:
        print('Epoch {} : {} / {}'.format(i, self.evaluate(test_data), n_test))
      else:
        print('Epoch {} finished'.format(j))
  def evaluate(self, test_data):
    test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    return sum(int(a == y) for (a, y) in test_results)
  def cost_derivative(self, output_activations, y):
    return (output_activations - y)

def sigmoid(x):
  return 1.0 / (1 + np.exp(-x))

def sigmoid_prime(x):
  return sigmoid(x) * (1 - sigmoid(x))