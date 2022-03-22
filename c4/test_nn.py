import mnist_loader
import nn

def main():
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

  net = nn.Network([784, 30, 10])

  net.gradient_descent(training_data, 10, 10, 0.1, test_data=test_data)

if __name__ == '__main__':
  main()