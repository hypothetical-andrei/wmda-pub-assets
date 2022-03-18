class MyClass():
  def __init__(self, content):
    self.content = content
  def printMe(self):
    print('content is ', self.content)

o = MyClass('somecontent')

o.printMe()