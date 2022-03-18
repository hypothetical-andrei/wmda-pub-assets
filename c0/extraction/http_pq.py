import requests
from pyquery import PyQuery as pq

def main():
  response = requests.get('http://andrei.ase.ro')
  html = response.text
  dom = pq(html)
  elements = dom('a')
  for element in elements:
    print(pq(element).text())
    print(pq(element).attr('href'))


if __name__ == '__main__':
  main()