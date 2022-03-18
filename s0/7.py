from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Firefox(executable_path='./geckodriver')
driver.get('https://www.imobiliare.ro/vanzare-apartamente/pitesti/prundu/apartament-de-vanzare-2-camere-XDCT1007Q?exprec=similare&rec_ref=home&sursa_rec=home&imoidviz=3395221058')

try:
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".pret-costuri__box--pret-cerut"))
    )
    print(element.text)
finally:
    driver.quit()
