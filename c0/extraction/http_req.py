import requests

def main():
  response = requests.get('http://andrei.ase.ro')
  print(response.text)

if __name__ == '__main__':
  main()