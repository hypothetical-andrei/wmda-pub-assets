import requests
import json

def main():
  template = 'http://sg.media-imdb.com/suggests/{first}/{start}.json'
  url = template.format(first = 'a', start = 'ab')
  response = requests.get(url)
  jsonpData = response.text
  jsonData = jsonpData[8:-1]
  recommendations = json.loads(jsonData)
  # with open('out.json', 'wt') as f:
  #   f.write(jsonData)
  for item in recommendations['d']:
    if item['s'].startswith('Act'):
      print(item['l'])

if __name__ == '__main__':
  main()