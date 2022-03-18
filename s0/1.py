def main():
  l = [1, 2, 3, 4, 5]
  print(sum([x for x in l if x % 2 == 0]))

if __name__ == '__main__':
  main()