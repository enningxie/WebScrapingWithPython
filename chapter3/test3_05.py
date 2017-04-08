# final
from urllib.request import urlopen
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
import datetime
import random


pages = set()
random.seed(datetime.datetime.now())


# 获取页面所有内链的列表
def getInternalLinks(bsObj, includerUrl):
    includerUrl = urlparse(includerUrl).scheme+"://"+urlparse(includerUrl).netloc
    internalLinks = []
    # 找出所有以‘/’开头的链接
    for link in bsObj.findAll("a", href=re.compile("^(/|.*"+includerUrl+")")):
        if link.attrs['href'] is not None:
            if link.attrs['href'] not in internalLinks:
                if link.attrs['href'].startswith("/"):
                    internalLinks.append(includerUrl+link.attrs['href'])
                else:
                    internalLinks.append(link.attrs['href'])
    return internalLinks


# 获取页面所有外链的链接
def getExternalLinks(bsObj, excludeUrl):
    externalLinks = []
    # 找出所有以"http"或"www"开头且不包括当前url的链接
    for link in bsObj.findAll("a", href=re.compile("^(http|www)((?!"+excludeUrl+").)*$")):
        if link.attrs['href'] is not None:
            if link.attrs['href'] not in externalLinks:
                externalLinks.append(link.attrs['href'])
    return externalLinks


def getRandomExternalLink(startingPage):
    html = urlopen(startingPage)
    bsObj = BeautifulSoup(html, "lxml")
    externalLinks = getExternalLinks(bsObj, urlparse(startingPage).netloc)
    if len(externalLinks) == 0:
        print("No external links, looking around the site for one.")
        domain = urlparse(startingPage).scheme+"://"+urlparse(startingPage).netloc
        internalLinks = getInternalLinks(bsObj, domain)
        return getRandomExternalLink(internalLinks[random.randint(0, len(internalLinks)-1)])
    else:
        return externalLinks[random.randint(0, len(externalLinks)-1)]


def followExternalOnly(startingSite):
    externalLink = getRandomExternalLink(startingSite)
    print("Random external link is:" + externalLink)
    followExternalOnly(externalLink)




# 收集网站上发现的所有外链列表
allExtLinks = set()
allIntLinks = set()


def getAllExternalLinks(siteUrl):
    html = urlopen(siteUrl)
    bsObj = BeautifulSoup(html, "lxml")
    domain = urlparse(siteUrl).scheme+"://"+urlparse(siteUrl).netloc
    internalLinks = getInternalLinks(bsObj, domain)
    externalLinks = getExternalLinks(bsObj, domain)

    for link in externalLinks:
        if link not in allExtLinks:
            allExtLinks.add(link)
            print(link)

    for link in internalLinks:
        if link not in internalLinks:
            allIntLinks.add(link)
            getAllExternalLinks(link)


followExternalOnly("http://www.cjlu.edu.cn")
allIntLinks.add("http://oreilly.com")
getAllExternalLinks("http://oreilly.com")