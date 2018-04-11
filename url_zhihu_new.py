# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 13:42:18 2017

@author: stan han
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 20:49:50 2016

@author: stan han
"""

'''
这是一个多线程的示范样例，应用于bt的，可以工作，
但是接口问题并没有处理完，因而，还需要做一些微调。
时间:2016.7.31 23:59


bt最新版
2016.8.1 23:33

已经结束，登陆已改用requests
2016.8.3
作者：stan

'''
import requests
import urllib.request

from bs4 import BeautifulSoup,Tag
import re
import PIL.Image
import time
import io
import codecs
def get_captcha(session,header) :
    url = 'https://www.zhihu.com/captcha.gif?r=%s&type=login'%str(int(time.time()*1000))
    pic = session.get(url,headers=header)
    pic = PIL.Image.open(io.BytesIO(pic.content))
    pic.show()
    return session


def login(session,header) :
    url = 'https://www.zhihu.com/'
    r = session.get(url,headers=header,verify=False)
    html = r.text
    #print(session.headers)
    bs = BeautifulSoup(html,'html.parser')
    xsrf = bs.find("input",{"name":"_xsrf"})
    print(xsrf["value"])
    return session,xsrf["value"]
'''
header = {
        'Host':'www.zhihu.com',
        'Connection':'keep-alive',
        'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.98 Safari/537.36',
        'Accept':'image/webp,image/*,*/*;q=0.8',
        'Referer':'https://www.zhihu.com/',
        'Accept-Encoding':'gzip, deflate, sdch, br',
        'Accept-Language':'zh-CN,zh;q=0.8'
}
'''
header = {
'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.98 Safari/537.36'
}


session = requests.Session()
url = "https://sz.xzl.anjuke.com/loupan/futian/"
page = session.get(url,headers=header)
page.encoding = page.apparent_encoding
with open('szanjuke.html','wb') as fi :
    fi.write(page.content)
# print(page.text)
# session,xsrf = login(session,header)
'''
# session = get_captcha(session,header)
captcha = input('输入验证码')
postd = {"code":captcha,
         "token":"xxxxx",
         "token2":"xxxxx",
         "submit":"提交"}

header1 = {
        'Host':'www.zhihu.com',
        'Connection':'keep-alive',
        'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.98 Safari/537.36',
        'Accept':'image/webp,image/*,*/*;q=0.8',
        'Referer':'https://www.zhihu.com/',
        'Accept-Encoding':'gzip, deflate, sdch, br',
        'Accept-Language':'zh-CN,zh;q=0.8',
        'X-Requested-With':'XMLHttpRequest'
}

url = 'https://www.zhihu.com/login/phone_num'
s = session.post(url,postd,headers=header)

print(s.text)
url = "https://www.zhihu.com/"
s = session.get(url,headers = header)
with open("zhihu.html","wb") as f :
    f.write(s.content)
print('done!')
'''