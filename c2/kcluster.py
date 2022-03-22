import distances
import random
from sklearn.datasets import load_iris
from PIL import Image, ImageDraw
from math import sqrt

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

def adjust(data, distance = distances.euclidean_inverse, rate = 0.01):
  n = len(data)
  realdist = [[distance(data[i], data[j]) for j in range(n)] for i in range(n)]
  fakedist = [[0.0 for j in range(n)] for i in range(n)]
  loc = [[random.random(), random.random()] for i in range(n)]
  lasterror = None
  for t in range(1000):
    for i in range(n):
      for j in range(n):
        fakedist[i][j] = sqrt(sum([pow(loc[i][x] - loc[j][x], 2) for x in range(len(loc[i]))]))
  
    grad = [[0.0, 0.0] for i in range(n)]
    totalerror = 0
    for i in range(n):
      for j in range(n):
        if i == j:
          continue
        errorterm = (fakedist[j][i] - realdist[j][i]) / realdist[j][i]
        grad[i][0] = errorterm * (loc[j][0] - loc[i][0]) / fakedist[i][j] 
        grad[i][1] = errorterm * (loc[j][1] - loc[i][1]) / fakedist[i][j]
        totalerror += errorterm
    if lasterror and lasterror < totalerror:
      break
    lasterror = totalerror
    for i in range(n):
      loc[i][0] -= grad[i][0] * rate
      loc[i][1] -= grad[i][1] * rate

  return loc

def draw2d(data, labels, jpeg='cluster.jpg'):
  img = Image.new('RGB', (2000, 2000), (255, 255, 255))
  draw = ImageDraw.Draw(img)
  for i in range(len(data)):
    x = (data[i][0] + 0.5) * 1000
    y = (data[i][1] + 0.5) * 1000
    draw.text((x, y), labels[i], (0, 0, 0))
  img.save(jpeg, 'JPEG')

def main():
  colnames, rownames, data = read_file('blogdata.txt')
  print(len(colnames))
  # clusters = kcluster(data, k = 3)
  # print(clusters)
  coords = adjust(data)
  draw2d(coords, rownames)
  # iris = load_iris()
  # data = iris.data
  # print(iris)
  # clusters = kcluster(data, k = 4)
  # for cluster in clusters:
  #   print(cluster)

if __name__ == '__main__':
  main()