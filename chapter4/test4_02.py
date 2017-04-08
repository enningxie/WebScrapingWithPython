# 收集wikipedia编辑页中的ip地址，并调用freegeoip.net中的API输出ip对应的国家
from urllib.request import urlopen
from bs4 import BeautifulSoup
import datetime
import re
import random
from urllib.error import HTTPError
import json


random.seed(datetime.datetime.now())


def getLinks(articleUrl):
    html = urlopen("http://en.wikipedia.org"+articleUrl)
    bsObj = BeautifulSoup(html, "lxml")
    return bsObj.find("div", {"id": "bodyContent"}).findAll("a", href=re.compile("^(/wiki/)((?!:).)*$"))


def getHistoryIPs(pageUrl):
    # 编辑历史页面的url链接格式是：
    # http://en.wikipedia.org/w/index.php?title=Title_in_URL&action=history
    pageUrl = pageUrl.replace("/wiki/", "")
    historyUrl = "http://en.wikipedia.org/w/index.php?title="+pageUrl+"&action=history"
    print("history url is :"+historyUrl)
    html = urlopen(historyUrl)
    bsObj = BeautifulSoup(html, "lxml")
    # 找出class属性是"mw-userlink mw-anonuserlink"的链接
    # 它们用ip地址代替用户名
    ipAddresses = bsObj.findAll("a", {"class": "mw-userlink mw-anonuserlink"})
    addressList = set()
    for ipAddress in ipAddresses:
        addressList.add(ipAddress.get_text())
    return addressList


def getCountry(ipAddress):
    try:
        response = urlopen("http://freegeoip.net/json/"+ipAddress).read().decode('utf-8')
    except HTTPError:
        return None
    responseJson = json.loads(response)
    return responseJson.get("country_name")


links = getLinks("/wiki/Python_(programming_language)")


while(len(links) > 0):
    for link in links:
        print("-----------------------")
        historyIPs = getHistoryIPs(link.attrs['href'])
        for historyIP in historyIPs:
            country_name = getCountry(historyIP)
            if country_name is not None:
                print(historyIP+"is from "+country_name)
    newLink = links[random.randint(0, len(links)-1)].attrs['href']
    links = getLinks(newLink)