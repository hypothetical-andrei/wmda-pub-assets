import requests
from pyquery import PyQuery as pq

def main():
  headers = {
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
  }
  response = requests.get('https://www.themoviedb.org/movie/772272-tall-girl-2', headers = headers)
  html = response.text
  dom = pq(html)
  elements = dom('li.profile')
  for element in elements:
    current = pq(element)
    recordType = current('p.character').text()
    if recordType == 'Director':
      name = current('p>a').text()
      print(name)
    


if __name__ == '__main__':
  main()