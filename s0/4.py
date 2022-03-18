import requests
from pyquery import PyQuery as pq

url = 'https://en.wikipedia.org/wiki/Romanian_leu'

response = requests.get(url)
html = response.text
dom = pq(html)
# elements = dom('.infobox tr:nth-child(5) .infobox-data')
# code = pq(elements[0]).text()

code = 'n/a'

elements = dom('.infobox tr')
for element in elements:
  element_dom = pq(element)
  label_element = element_dom('.infobox-label')
  if pq(label_element).text() == 'Code':
    code = element_dom('.infobox-data').text()
    break

print(code)