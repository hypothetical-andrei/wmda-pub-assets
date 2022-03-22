from sklearn.naive_bayes import GaussianNB
import util

def main():
  messages = util.getSpamFeaturesSk('spambase')
  samples = messages[100:]
  tests = messages[:100]
  sampleFeatures = [item['features'] for item in samples]
  sampleOutcomes = [item['outcome'] for item in samples]
  gnb = GaussianNB()
  classifier = gnb.fit(sampleFeatures, sampleOutcomes)
  testFeatures = [item['features'] for item in tests]
  testOutcomes = [item['outcome'] for item in tests]
  predicted = classifier.predict(testFeatures)
  correct = len([item for item in zip(testOutcomes, predicted) if item[0] == item[1]])
  print('correct : ', correct, ' | incorrect : ', 100 - correct)

if __name__ == '__main__':
  main()