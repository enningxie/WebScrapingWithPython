# 解析json数据
import json
from urllib.request import urlopen


def getCountry(ipAddress):
    response = urlopen("http://freegeoip.net/json/"+ipAddress).read().decode('utf-8')
    responseJson = json.loads(response)
    return responseJson.get("country_name")


print(getCountry("103.192.224.159"))