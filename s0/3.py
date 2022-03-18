import json

def convert_file(source):
  results = []
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
      results.append(result)
  json_content = json.dumps(results)
  with open('test.json', 'wt') as f:
    f.write(json_content)

def main():
  sample_file = 'test.csv'
  convert_file(sample_file)

if __name__ == '__main__':
  main()