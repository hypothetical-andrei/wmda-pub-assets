def iter_file(source):
  with open(source) as f:
    lines = f.readlines()
    heading = lines[0]
    heading_items = heading.strip().split(',')
    for line in lines[1:]:
      items = line.strip().split(',')
      result = {
        heading_items[0]: items[0],
        heading_items[1]: items[1],
        heading_items[2]: items[2]
      }
      yield result

def main():
  sample_file = 'test.csv'
  for item in iter_file(sample_file):
    print(item)

if __name__ == '__main__':
  main()