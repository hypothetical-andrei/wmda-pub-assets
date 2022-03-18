from math import sqrt

def euclidean(v1, v2):
  sum_squares = sum([(x - y) ** 2 for (x, y) in zip(v1, v2)])
  return sqrt(sum_squares)

def main():
  sample_vector1 = [1,2,3,4,5]
  sample_vector2 = [3,4,5,6,7]
  print(euclidean(sample_vector1, sample_vector2))

if __name__ == '__main__':
  main()