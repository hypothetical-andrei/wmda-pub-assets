import requests
from pyquery import PyQuery as pq

url = 'https://www.themoviedb.org/movie/553-dogville'

headers = {
  'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
}

response = requests.get(url, headers = headers)
html = response.text
# print(html)
dom = pq(html)
elements = dom('section.facts p')

element_contents = list(map(lambda e: pq(e).text() , elements))

for content in element_contents:
  items = content.split('\n')
  if items[0] == 'Budget':
    print(items[1])
    continue
  if items[0] == 'Revenue':
    print(items[1])
    continue
  