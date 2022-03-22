def loadCarData(file = 'car.data'):
  results = []
  with open(file) as f:
    for line in f.readlines():
      items = line.strip().split(',')
      results.append(items)
  return results

def main():
  results = loadCarData()
  print(results[:5])

if __name__ == '__main__':
  main()