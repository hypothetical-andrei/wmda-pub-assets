from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier

digits = datasets.load_digits()

num_samples = len(digits.images)

data = digits.images.reshape((num_samples, -1))

classifier = MLPClassifier(hidden_layer_sizes=(30))

classifier.fit(data[:int(num_samples * 0.9)], digits.target[:int(num_samples * 0.9)])

expected = digits.target[int(num_samples * 0.9):]

predicted = classifier.predict(data[int(num_samples * 0.9):])

print('Classification report')
print(metrics.classification_report(expected, predicted))

print('Confusion matrix')
print(metrics.confusion_matrix(expected, predicted))