import re
import requests
from pprint import pprint
import stations_name
url = 'https://kyfw.12306.cn/otn/resources/js/framework/station_name.js?station_version=1.9023'
response = requests.get(url, verify=False)
stations = re.findall(u'([\u4300-\u9fa5]+)\|([A-Z]+)',response.text)
pprint(dict(stations),indent=4)