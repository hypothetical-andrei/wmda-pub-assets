from sklearn import cluster
import numpy as np
import itertools

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


def print_cluster(clusters, current_id, initial_value, labels = None, n = 0):
  for i in range(n):
    print(' ', end = '')
  if current_id < initial_value:
    if labels == None:
      print(current_id)
    else:
      print(labels[current_id])
  else:
    print('-')
    current_cluster = [c for c in clusters if current_id == c['id']][0]
    print_cluster(clusters, current_cluster['left'], initial_value, labels = labels, n = n + 1)
    print_cluster(clusters, current_cluster['right'], initial_value, labels = labels, n = n + 1)

def main():
  colsname, rownames, data = read_file('blogdata.txt')
  numpy_data = np.array(data)
  clusterer = cluster.AgglomerativeClustering(compute_full_tree = True, n_clusters = 2, linkage = 'complete')
  model = clusterer.fit(numpy_data)
  print(model.children_)
  it = itertools.count(numpy_data.shape[0])
  v = [{'id': next(it), 'left': x[0], 'right':x[1]} for x in model.children_]
  print_cluster(v, v[-1]['id'], numpy_data.shape[0], labels = rownames)

if __name__ == '__main__':
  main()