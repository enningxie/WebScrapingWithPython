# 正则表达式和BeautifulSoup
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re


html = urlopen("http://pythonscraping.com/pages/page3.html")
bsObj = BeautifulSoup(html, "lxml")
images = bsObj.findAll("img", {"src": re.compile("\.\.\/img\/gifts\/img.*\.jpg")})
for image in images:
    print(image["src"])

# 对于一个标签对象，myTag.attrs获取它的全部属性，返回的是一个字典对象
# lxml 库，可以用来解析HTML和XML文档，以非常底层的实现而闻名于世，大部分源代码是用C写的，处理绝大多数HTML文档时速度都非常快。