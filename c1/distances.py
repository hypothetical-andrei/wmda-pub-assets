import data_sample
from math import sqrt

def euclidean(prefs, me, other):
  shared = {}
  for item in prefs[me]:
    if item in prefs[other]:
      shared[item] = 1
  if len(shared) == 0:
    return 0
  square_sum = sum([pow(prefs[me][item] - prefs[other][item], 2) for item in prefs[me] if item in prefs[other]])
  return 1 / (1 + sqrt(square_sum))

def main():
  data = data_sample.critics
  print(euclidean(data, 'Lisa Rose', 'Gene Seymour'))

if __name__ == '__main__':
  main()