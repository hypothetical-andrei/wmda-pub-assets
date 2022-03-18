import distances
import random
from sklearn.datasets import load_iris

def read_file(filename):
  with open(filename) as f:
    lines = f.readlines()
  colnames = lines[0].strip().split('\t')[1:]
  rownames = []
  data = []
  for line in lines[1:]:
    parts = line.strip().split('\t')
    rownames.append(parts[0])
    data.append([float(x) for x in parts[1:]])
  return colnames, rownames, data

def kcluster(rows, distance = distances.euclidean, k = 4):
  last_matches = None
  intervals = [(min([row[i] for row in rows]), max([row[i] for row in rows])) for i in range(len(rows[0]))]
  clusters = [[random.random() * (intervals[i][1] - intervals[i][0]) + intervals[i][0] for i in range(len(rows[0]))] for j in range(k)]
  MAX_ITER = 100
  for t in range(MAX_ITER):
    # print('iteration ', t)
    best_matches = [[] for i in range(k)]
    for j in range(len(rows)):
      current_row = rows[j]
      best_match = 0
      for i in range(k):
        d = distance(clusters[i], current_row)
        if d < distance(clusters[best_match], current_row):
          best_match = i
      best_matches[best_match].append(j)
    if best_matches == last_matches:
      break
    for i in range(k):
      avgs = [0.0] * len(rows[0])
      if len(best_matches[i]) > 0:
        for row_id in best_matches[i]:
          for j in range(len(rows[row_id])):
            avgs[j] += rows[row_id][j]
        for j in range(len(avgs)):
          avgs[j] /= len(best_matches[i])
        clusters[i] = avgs
  return best_matches

def main():
  # colnames, rownames, data = read_file('blogdata.txt')
  # clusters = kcluster(data, k = 3)
  # print(clusters)

  iris = load_iris()
  data = iris.data
  print(iris)
  clusters = kcluster(data, k = 4)
  for cluster in clusters:
    print(cluster)

if __name__ == '__main__':
  main()