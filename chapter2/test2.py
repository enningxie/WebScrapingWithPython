# 复杂HTML解析
from urllib.request import urlopen
from bs4 import BeautifulSoup


html = urlopen("http://pythonscraping.com/pages/warandpeace.html")
bsObj = BeautifulSoup(html, "lxml")

namelist = bsObj.findAll("span", {"class": "green"})
for name in namelist:
    print(name.get_text())