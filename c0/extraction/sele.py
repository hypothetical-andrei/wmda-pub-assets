from selenium import webdriver
from selenium.webdriver.common.by import By

def main():
  driver = webdriver.Firefox(executable_path='./geckodriver')
  driver.set_window_size(800, 600)
  driver.get('http://andrei.ase.ro')
  element = driver.find_element(By.CSS_SELECTOR, 'a[href^=wmda]')
  print(element.text)
  driver.quit()

if __name__ == '__main__':
  main()