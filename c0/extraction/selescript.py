from selenium import webdriver
from selenium.webdriver.common.by import By

def main():
  with open('selescript.js') as f:
    script = f.read()
  driver = webdriver.Firefox(executable_path='./geckodriver')
  driver.set_window_size(800, 600)
  driver.get('http://andrei.ase.ro')
  result = driver.execute_script(script)
  print(result)
  driver.quit()

if __name__ == '__main__':
  main()