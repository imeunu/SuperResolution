import os
import time
import urllib.request

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def crawl(chromedriver, url, save_dir):
    driver = webdriver.Chrome(chromedriver)
    driver.get(url)
    for i in range(9,24,2):
        driver.find_elements_by_tag_name('a')[i].click()
        for j in range(25):
            driver.find_elements_by_tag_name('img')[j].click()
            imgurl = driver.find_element_by_tag_name('img').get_attribute('src')
            path = os.path.join(save_dir,imgurl.split('/')[-1])
            urllib.request.urlretrieve(imgurl,path)
            time.sleep(0.5)
            driver.back()
        driver.back()
    driver.close()

if __name__ == '__main__':
    chromedriver = 'C:\\Users\\eunwoo\\Desktop\\Code\\Study\\selenium\\chromedriver.exe'
    url = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/dataset/images.html'
    save_dir = 'C:\\Users\\eunwoo\\Desktop\\BSD'
    crawl(chromedriver, url, save_dir)