from sample import sample
from math import log
from PIL import Image, ImageDraw
import util

class DecisionNode:
  def __init__(self, col = -1, value = None, results = None, falseSubtree = None, trueSubtree = None):
    self.col = col
    self.value = value
    self.results = results
    self.falseSubtree = falseSubtree
    self.trueSubtree = trueSubtree

def uniquecounts(rows):
  results = {}
  for row in rows:
    outcome = row[-1]
    results.setdefault(outcome, 0)
    results[outcome]  += 1
  return results

def entropy(rows):
  log2 = lambda x: log(x)/log(2)
  results = uniquecounts(rows)
  h = 0.0
  for result, count in results.items():
    p = float(count / len(rows))
    h -= p * log2(p)
  return h

def divideSet(rows, column, value):
  splitFunction = None
  if isinstance(value, int) or isinstance(value, float):
    splitFunction = lambda row: row[column] >= value
  else:
    splitFunction = lambda row: row[column] == value
  trueSet = [row for row in rows if splitFunction(row)]
  falseSet = [row for row in rows if not splitFunction(row)]
  return (falseSet, trueSet)

def buildTree(rows, scoref = entropy):
  if len(rows) == 0:
    return DecisionNode()
  currentScore = scoref(rows)
  bestGain = 0.0
  bestCriterion = None
  bestSets = None
  columnCount = len(rows[0]) - 1
  for column in range(0, columnCount):
    columnValues = {}
    for row in rows:
      columnValues[row[column]] = 1
      for value in columnValues.keys():
        (falseSet, trueSet) = divideSet(rows, column, value)
        p = float(len(falseSet) / len(rows))
        gain = currentScore - p * scoref(falseSet) - (1 - p) * scoref(trueSet)
        if gain > bestGain and len(falseSet) > 0:
          bestGain = gain
          bestCriterion = (column, value)
          bestSets = (falseSet, trueSet)
  if bestGain > 0.0:
    falseBranch = buildTree(bestSets[0])
    trueBranch = buildTree(bestSets[1])
    return DecisionNode(col = bestCriterion[0], value = bestCriterion[1], trueSubtree = trueBranch, falseSubtree = falseBranch)
  else:
    return DecisionNode(results = uniquecounts(rows))

def classify(item, node):
  if node.results != None:
    return node.results
  else:
    value = item[node.col]
    branch = None
    if isinstance(value, int) or isinstance(value, float):
      print('decision column ', node.col, ' >= value ', value)
      if value >= node.value:
        branch = node.trueSubtree
      else:
        branch = node.falseSubtree
    else:
      print('decision column ', node.col, ' == value ', value)
      if value == node.value:
        branch = node.trueSubtree
      else:
        branch = node.falseSubtree
  return classify(item, branch)

def printTree(node, indent = ''):
  if node.results != None:
    print(node.results)
  else:
    print(node.col, ':', node.value, '?')
    print(indent, 'True -> ', end='')
    printTree(node.trueSubtree, indent=indent + '   ')
    print(indent, 'False ->', end='')
    printTree(node.falseSubtree, indent=indent + '   ')

def getWidth(node):
  if node.trueSubtree == None and node.falseSubtree == None:
    return 1
  else:
    return getWidth(node.trueSubtree) + getWidth(node.falseSubtree)

def getHeight(node):
  if node.trueSubtree == None and node.falseSubtree == None:
    return 0
  else:
    return 1 + max(getHeight(node.trueSubtree), getHeight(node.falseSubtree))

def drawTree(tree, file='tree.jpg'):
  w = getWidth(tree) * 100
  h = getHeight(tree) * 100 + 120
  img = Image.new('RGB', (w, h), (255, 255, 255))
  draw = ImageDraw.Draw(img)
  drawNode(draw, tree, w / 2, 20)
  img.save(file, 'JPEG')

def drawNode(draw, node, x, y):
  if node.results != None:
    results = ['{outcome}:{count}'.format(outcome = k, count = v) for k, v in node.results.items()]
    text = ', '.join(results)
    draw.text((x - 20, y), text, (0, 0, 0))
  else:
    falseWidth = getWidth(node.falseSubtree) * 100
    trueWidth = getWidth(node.trueSubtree) * 100
    left = x - (trueWidth + falseWidth) / 2
    right = x + (trueWidth + falseWidth) / 2
    draw.text((x - 20, y - 10), str(node.col) + ':' + str(node.value), (0, 0, 0))
    draw.line((x, y, left + falseWidth / 2, y + 100), fill = (255, 0, 0))
    draw.line((x, y, right - trueWidth / 2, y + 100), fill = (0, 255, 0))
    drawNode(draw, node.falseSubtree, left + falseWidth / 2, y + 100)
    drawNode(draw, node.trueSubtree, right - trueWidth / 2, y + 100)

# https://archive.ics.uci.edu/ml/machine-learning-databases/car/
def main():
  # data = sample
  data = util.loadCarData()
  # tree = buildTree(sample)
  tree = buildTree(data)
  # testSample = ['digg', 'USA', 'yes', 19]
  # result = classify(testSample, tree)
  # print(result)
  printTree(tree)
  drawTree(tree)

if __name__  == '__main__':
  main()