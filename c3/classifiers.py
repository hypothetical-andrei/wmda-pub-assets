import util
import random

class Classifier:
  def __init__(self, getFeatures):
    self.getFeatures = getFeatures
    self.cc = {}
    self.fc = {}
  def incf(self, f, cat):
    self.fc.setdefault(f, {})
    self.fc[f].setdefault(cat, 0)
    self.fc[f][cat] += 1
  def incc(self, cat):
    self.cc.setdefault(cat, 0)
    self.cc[cat] += 1
  def totalCount(self):
    return sum(self.cc.values())
  def catCount(self, cat):
    if cat in self.cc:
      return self.cc[cat]
    return 0
  def fcount(self, f, cat):
    if f in self.fc and cat in self.fc[f]:
      return self.fc[f][cat]
    return 0
  def categories(self):
    return self.cc.keys()
  def train(self, item, cat):
    features = self.getFeatures(item)
    for f in features:
      self.incf(f, cat)
    self.incc(cat)
  def fprob(self, f, cat):
    if self.catCount(cat) != 0:
      return self.fcount(f, cat) / self.catCount(cat)
    return 0

class NaiveBayesClassifier(Classifier):
  def __init__(self, getFeatures):
    Classifier.__init__(self, getFeatures)
    self.thresholds = {}
  def setThreshold(self, cat, t):
    self.thresholds[cat] = t
  def prob(self, item, cat):
    catProb = self.catCount(cat) / self.totalCount()
    messageProbability = self.messageProbability(item, cat)
    return catProb * messageProbability
  def messageProbability(self, item, cat):
    features = self.getFeatures(item)
    p = 1
    for f in features:
      p *= self.fprob(f, cat)
    return p
  def classify(self, item):
    maxProbability = 0
    bestCat = None
    for cat in self.categories():
      prob = self.prob(item, cat)
      if prob > maxProbability:
        maxProbability = prob
        bestCat = cat
    return bestCat

def main():
  # messages = util.getSpamFeatures('spambase')
  # random.shuffle(messages)
  # samples = messages[100:]
  # tests = messages[:100]
  # classifier = NaiveBayesClassifier(util.getFeatures)
  # for item in samples:
  #   classifier.train(item, item['outcome'])
  # correct = 0
  # for item in tests:
  #   result = classifier.classify(item)
  #   if result == item['outcome']:
  #     correct += 1
  # print('correct: ', correct, ' |  incorrect : ', 100 - correct)
  data = util.loadCarData('car.data')
  # random.shuffle(data)
  samples = data[100:]
  tests = data[:100]
  classifier = NaiveBayesClassifier(util.getFeatures)
  for item in samples:
    classifier.train(item, item['outcome'])
  correct = 0
  for item in tests:
    result = classifier.classify(item)
    if result == item['outcome']:
      correct += 1
  print('correct: ', correct, ' |  incorrect : ', 100 - correct)

if __name__ == '__main__':
  main()

