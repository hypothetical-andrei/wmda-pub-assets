def myGenerator():
  for item in range(0, 10):
    yield item

for item in myGenerator():
  print(item)