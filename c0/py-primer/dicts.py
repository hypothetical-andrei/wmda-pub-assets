myDict = {
  'a': 1,
  'b': 2,
  'c': 3
}

print(myDict)

for k, v in myDict.items():
  print('myDict[{key}]={value}'.format(key = k, value = v))

if 'a' in myDict:
  myDict['a'] =  10

print(myDict)

myTuple = (1, 'a')

print(myTuple)