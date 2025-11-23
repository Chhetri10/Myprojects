import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from time import sleep
from selenium.webdriver.common.by import By

executable_path = 'C:/My projects/chromedriver-win64/chromedriver-win64/chromedriver.exe'

# Get absolute path to the 'data' folder in the current directory
download_dir = os.path.join(os.getcwd(), 'data')

# Make sure the 'data' directory exists
os.makedirs(download_dir, exist_ok=True)

# Set Chrome preferences for downloading
chrome_options = Options()
chrome_prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "directory_upgrade": True,
    "safebrowsing.enabled": True
}
chrome_options.add_experimental_option("prefs", chrome_prefs)

# Start Chrome with options
cService = ChromeService(executable_path=executable_path)
driver = webdriver.Chrome(service=cService, options=chrome_options)
# Open URL
driver.get(
    'https://omitnomis.github.io/ShareSansarScraper/preview.html')

# Enter text
# driver.find_element_by_id('textbox').send_keys("Hello world")

# Generate Text File
# driver.find_element_by_id('createTxt').click()
sleep(10)
# Click on Download Button
driver.find_element(By.XPATH, '//*[@id="main-content"]/section/div[1]/button').click()
sleep(30)
driver.quit()
