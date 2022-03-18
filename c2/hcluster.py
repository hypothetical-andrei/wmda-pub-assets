import distances
from PIL import Image, ImageDraw

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

class BiCluster:
  def __init__(self, vec, left = None, right = None, distance = 0.0, id = None):
    self.left = left
    self.right = right
    self.vec = vec
    self.id = id
    self.distance = distance

def hcluster(rows, distance = distances.euclidean):
  distances = {}
  current_cluster_id = -1
  clusters = [BiCluster(rows[i], id = i) for i in range(len(rows))]
  while len(clusters) > 1:
    # print('current cluster vector length', len(clusters))
    lowest_pair = (0, 1)
    closest = distance(clusters[0].vec, clusters[1].vec)
    for i in range(len(clusters)):
      for j in range(i+1, len(clusters)):
        if (clusters[i].id, clusters[j].id) not in distances:
          distances[(clusters[i].id, clusters[j].id)] = distance(clusters[i].vec, clusters[j].vec)
        current_distance = distances[(clusters[i].id, clusters[j].id)]
        if current_distance < closest:
          closest = current_distance
          lowest_pair = (i, j)
    merge_vector = [(clusters[lowest_pair[0]].vec[i] + clusters[lowest_pair[1]].vec[i]) / 2.0 for i in range(len(clusters[0].vec))]
    merged_cluster = BiCluster(merge_vector, left = clusters[lowest_pair[0]], right = clusters[lowest_pair[1]], distance = closest, id = current_cluster_id)
    current_cluster_id -= 1
    del clusters[lowest_pair[1]]
    del clusters[lowest_pair[0]]
    clusters.append(merged_cluster)
  return clusters[0]

def print_cluster(cluster, labels = None, n = 0):
  for i in range(n):
    print(' ', end='')
  if cluster.id < 0:
    print('+')
    # print(cluster.id)
  else:
    if labels == None:
      print(cluster.id)
    else:
      print(labels[cluster.id])
  if cluster.left != None:
    print_cluster(cluster.left, labels = labels, n = n + 1)
  if cluster.right != None:
    print_cluster(cluster.right, labels = labels, n = n + 1)

def get_depth(cluster):
  if cluster.left == None and cluster.right == None:
    return 0
  return max(get_depth(cluster.left), get_depth(cluster.right)) + cluster.distance

def get_height(cluster):
  if cluster.left == None and cluster.right == None:
    return 1
  return get_height(cluster.left) + get_height(cluster.right)

def draw_node(draw, cluster, x, y, scaling, labels):
  if cluster.id < 0:
    hl = get_height(cluster.left) * 20
    hr = get_height(cluster.right) * 20
    top = y - (hl + hr) / 2
    bottom = y + (hl + hr) / 2
    line_length = cluster.distance * scaling
    draw.line((x, top + hl / 2, x, bottom - hr / 2), fill = (255, 0, 0))
    draw.line((x, top + hl / 2, x + line_length, top + hl / 2), fill = (255, 0, 0))
    draw.line((x, bottom - hr / 2, x + line_length, bottom - hr / 2), fill = (255, 0, 0))
    draw_node(draw, cluster.left, x + line_length, top + hl / 2, scaling, labels)
    draw_node(draw, cluster.right, x + line_length, bottom - hr / 2, scaling, labels)
  else:
    draw.text((x + 5, y - 7), labels[cluster.id], (0, 0, 0))

def draw_dendrogram(cluster, labels, jpeg='clusters.jpg'):
  h = get_height(cluster) * 20
  w = 1200
  depth = get_depth(cluster)
  scaling = float(w - 300) / depth
  img = Image.new('RGB', (w, h), (255, 255, 255))
  draw = ImageDraw.Draw(img)
  draw.line((0, h / 2, 10, h / 2), fill = (255, 0, 0))
  draw_node(draw, cluster, 10, h / 2, scaling, labels)
  img.save(jpeg, 'JPEG')

def main():
  colnames, rownames, data = read_file('blogdata.txt')
  # print(colnames[:3])
  # print(rownames[:3])
  # print(data[1][3:9])
  root = hcluster(data)
  # print_cluster(root, labels=rownames)
  draw_dendrogram(root, labels=rownames)

if __name__ == '__main__':
  main()