# .children标签的使用.descendants的区别
from urllib.request import urlopen
from bs4 import BeautifulSoup


html = urlopen("http://pythonscraping.com/pages/page3.html")
bsObj = BeautifulSoup(html.read(), "lxml")

for child in bsObj.find("table", {"id": "giftList"}).descendants:
    print(child)