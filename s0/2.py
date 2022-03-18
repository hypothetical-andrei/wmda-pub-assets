def my_filter(input_list, predicate):
  return list([e for e in input_list if predicate(e)])

def main():
  sample_list = [{
    "name": "jim",
    "age": 22
  }, {
    "name": "tim",
    "age": 25
  }, {
    "name": "jane",
    "age": 20
  }]
  filtered_list = my_filter(sample_list, lambda e: e['age'] > 22)
  print(filtered_list)

if __name__ == '__main__':
  main()