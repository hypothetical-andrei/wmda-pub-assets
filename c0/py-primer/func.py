def myFunction(a, b, c, d = 10, e = 11):
  return a + b + c + d + e

print(myFunction(1, 2, 3))
print(myFunction(1, 2, 3, 4, 5))
print(myFunction(1, 2, 3, e = 5))

l = [1, 2, 3, 4, 5]

def myTransformation(x):
  return x * 2

mappedList = map(myTransformation, l)

print(list(mappedList))

print(list(map(lambda x: x * 2, l)))

comprehensionResult = [x * 2 for x in l if x > 3]

print(comprehensionResult)