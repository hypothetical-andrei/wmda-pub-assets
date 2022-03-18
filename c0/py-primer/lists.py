myList = ['a', 'b', 'c', 'd']
print(myList)
myList.append('e')
print(myList)
myList.extend(['f', 'g'])
print(myList)
print(myList[0])
print(myList[1:4])
print(myList[1:])
print(myList[:4])
print(myList[:])
print(myList[1:-1])

print('iteration')
for item in myList:
  print(item, end=' | ')

print()

print('enumeration')
for index, item in enumerate(myList):
  print('myList[{index}]={item}'.format(index = index, item = item))

numbers = []
for i in range(0, 10):
  numbers.append(i)

print(numbers)