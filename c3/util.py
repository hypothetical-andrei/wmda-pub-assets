def getSpamFeatures(filename, threshold = 0.1):
  with open(filename + '.names') as f:
    names = [line.split(':')[0] for line in f.readlines()]
  results = []
  with open(filename + '.csv') as f:
    for line in f.readlines():
      items = line.split(',')
      items = [float(item.strip()) for item in items]
      classification = items[-1]
      wordFeatures = items[:-10]
      features = []
      for index, item in enumerate(wordFeatures):
        if item > threshold:
          features.append(names[index])
      result = {
        'features':  features,
        'outcome': 'bad' if classification == 1 else 'good'
      }
      results.append(result)
  return results

def getSpamFeaturesSk(filename):
  results = []
  with open(filename + '.csv') as f:
    for line in f.readlines():
      items = line.split(',')
      items = [float(item.strip()) for item in items]
      classification = items[-1]
      wordFeatures = items[:-1]
      result = {
        'features': wordFeatures,
        'outcome': 'bad' if classification == 1 else 'good'
      }
      results.append(result)
  return results

def getFeatures(item):
  return item['features']

def loadCarData(filename):
  results = []
  with open(filename) as f:
    for line in f.readlines():
      items = line.split(',')
      items = [item.strip() for item in items]
      result = {
        'features': items[:-1],
        'outcome': items[-1]
      }
      results.append(result)
  return results

def main():
  results = getSpamFeatures('spambase')
  print(results[0:3])

if __name__ == '__main__':
  main()