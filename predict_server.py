# -*- coding: utf-8 -*-
from bottle import *
from sklearn import ensemble
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import json
from MySqlConn import Mysql  
from _sqlite3 import Row 
import math
from sklearn import ensemble
import os
from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

import sys
from itertools import chain, combinations
from collections import defaultdict

def allow_cross_domain(fn):  
    def _enable_cors(*args, **kwargs):  
        #set cross headers  
        response.headers['Access-Control-Allow-Origin'] = '*'  
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,DELETE,PUT,OPTIONS'  
        allow_headers = 'Referer, Accept, Origin, User-Agent,X-Requested-With,Content-Type'  
        response.headers['Access-Control-Allow-Headers'] = allow_headers       
        if request.method != 'OPTIONS':  
            # actual request; reply with the actual response  
            return fn(*args, **kwargs)      
    return _enable_cors  

def getNewLatLng(lat,lng,radius):
    degree = (24901*1609)/360
    dpmLat = 1/degree
    rLat = dpmLat*radius
    minLat = lat - rLat
    maxLat = lat + rLat

    mpdLng = degree*math.cos(lat*(math.pi/180))
    dpmLng = 1/mpdLng
    rLng = dpmLng*radius
    minLng = lng - rLng
    maxLng = lng + rLng
    return {'minLat':minLat,'maxLat':maxLat,'minLng':minLng,'maxLng':maxLng};

def permutate(file,rate):#训练数据 测试数据划分比例
    a = np.loadtxt(file,delimiter=',')
    list_a = [i for i in a]
    dict_a = {}
    n = a.shape[0]             # n行m列
    m = a.shape[1]
    for i in range(n):
        dict_a[i] = list_a[i]       # 将矩阵逐行存储在字典里
    # 生成n个随机序列存放在字典中，然后将序列排序，排序后的索引放于list[b]中,最终生成列表类似[3,1,2,0,4]
    rand_number = np.random.rand(n)
    dict_rand = {}
    for i in range(n):
        dict_rand[rand_number[i]] = i
    list_rand = sorted(dict_rand.items())
    list_num = [i[1] for i in list_rand]
    list_b = []
    # 按照生成列表依次提取csv相应行的数据，如[2,0,1]，则生成list_b = [dict_a[2],dict_a[0],dict_a[1]]
    for j in range(n):
        q = list_num[j]
        list_b.append(dict_a[q])
    # 将list_b转为矩阵格式
    tmp1 = math.floor(n*rate)
    train = np.zeros((tmp1, m))
    for i in range(0,tmp1):
        train[i, :] = list_b[i]
    test = np.zeros((n-tmp1,m)) 
    for i in range(n-tmp1):
        test[i, :] = list_b[tmp1+i]
    return {'train':train,'test':test,'file_rows':n,'test_rows':n-tmp1}

def prebsum():
    mysql = Mysql()
    sql = 'SELECT bsum,node_in_prer,firm_in_prer,house_in_prer,region_index,bank_index FROM pre_r WHERE region_index=2 or region_index=7 or region_index=8'
    res = mysql.getAll(sql)
    mysql.dispose()
    a = []
    for x in res:
        b=[]
        b.append(x['bsum'])
        b.append(x['node_in_prer'])
        b.append(x['firm_in_prer'])
        b.append(x['house_in_prer'])
        b.append(x['region_index'])
        b.append(x['bank_index'])
        a.append(b)    
    np.savetxt('to_predict_bsum.csv',a,delimiter=',')

    mse_arr=[]
    b = permutate('pre_bsum_train.csv',0.98)
    file_rows = b['file_rows']-b['test_rows']
    train_rows = math.floor(file_rows*0.9)
    np.savetxt('train.csv', b['train'], delimiter = ',')
    np.savetxt('test.csv', b['test'], delimiter = ',')
        
    for time in range(50):
        os.chdir("C:/Users/yufeng/Desktop/python")
        t = permutate('train.csv',0.98)
        np.savetxt('train_t.csv', t['train'], delimiter = ',')
        df = pd.read_csv('train_t.csv', header=0, encoding='utf-8')
        y_train,x_train = df.ix[0:train_rows,0:1],df.ix[0:train_rows,1:]
        y_test,x_test = df.ix[train_rows:,0:1],df.ix[train_rows:,1:]
        params = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 2,
                'learning_rate': 0.01, 'loss': 'ls'}
        gbr = ensemble.GradientBoostingRegressor(**params) 
        gbr.fit(x_train, y_train)
        os.chdir("/home/czhou/python/model")
        joblib.dump(gbr, "pre_bsum_model"+ str(time) + ".m")
        y_pre = gbr.predict(x_test)
        y_test = np.array(y_test)
        m=[]
        n=[]
        for i,j in enumerate(y_pre):
            m.append(y_pre[i]/(y_test[i][0]+y_pre[i])) 
            n.append(y_test[i][0]/(y_test[i][0]+y_pre[i]))
        mse = mean_squared_error(n,m)
        mse_arr.append(mse)

    a = mse_arr.index(min(mse_arr))
    os.chdir("/home/czhou/python")
    df1 = pd.read_csv('to_predict_bsum.csv', header=0, encoding='utf-8')
    y_test,x_test = df1.ix[0:,0:1],df1.ix[0:,1:]
    os.chdir("/home/czhou/python/model")
    gbr = joblib.load("pre_bsum_model"+ str(a)+".m")
    print("train_model"+ str(a)+".m")
    joblib.dump(gbr, "pre_bsum_model.m")
    y_pre1 = gbr.predict(x_test)
    index=np.arange(1,24,1)
    m=[]
    n=[]
    y_test = np.array(y_test)
    for i,j in enumerate(y_pre1):
        m.append(y_pre1[i]/(y_test[i][0]+y_pre1[i])) 
        n.append(y_test[i][0]/(y_test[i][0]+y_pre1[i]))
    mse = mean_squared_error(n,m)
    print("MSE: %.4f" % mse)
    # plt.plot(index,y_pre1,'r-',label='predict')
    # plt.plot(index,y_test,'b-',label='real')
    # plt.legend(loc='upper right')
    # plt.show()


def getCoverR():
    mysql = Mysql()
    nameSql = 'SELECT name FROM bankdata_copy'
    res1 = mysql.getAll(nameSql)
    for x in res:
        crSql = 'SELECT f_s_dis,flng,flat,slng,slat FROM original_data WHERE station_name=' + "'" + x['name'] + "'"
        res2 = mysql.getAll(crSql)
        nodeRadius = 0
        sorted(res2.items(),key=lambda item:item[0])
        if len(res2) > 0:
            nodeRadius = res2[math.floor(len(res2)*0.2)]['f_s_dis']
            tmp = res2[math.floor(len(res2)*0.2)]
            k = getNewLatLng(tmp['slat'],tmp['slng'],nodeRadius)
            xqSql = 'SELECT total_house from fdd_xq WHERE lat >' + str(k['minLat']) + ' AND lat <' + str(k['maxLat']) + ' AND lng >' + str(k['minLng']) + ' AND lng <' + str(k['maxLng'])
            res3 = mysql.getAll(xqSql)
            total_house = 0
            for m in res3:
                total_house +=m['total_house']
            print(total_house)

            xzlSql = 'SELECT area from biz_bld WHERE lat >' + str(k['minLat']) + ' AND lat <' + str(k['maxLat']) + ' AND lng >' + str(k['minLng']) + ' AND lng <' + str(k['maxLng'])
            res4 = mysql.getAll(xzlSql)
            total_area = 0
            for m in res3:
                total_area +=m['area']
            print(total_area)

            netSql = 'SELECT COUNT(name) AS data from bankdata_copy WHERE lat >' + str(k['minLat']) + ' AND lat <' + str(k['maxLat']) + ' AND lng >' + str(k['minLng']) + ' AND lng <' + str(k['maxLng'])
            r1 = mysql.getOne(netSql)
            print(r1)

            firmSql = 'SELECT person_num from firm_info WHERE flat >' + str(k['minLat']) + ' AND flat <' + str(k['maxLat']) + ' AND flng >' + str(k['minLng']) + ' AND flng <' + str(k['maxLng'])
            r2 = mysql.getAll(firmSql)
            person_num = 0
            for m in r2:
                person_num +=m['person_num']
            print(person_num)

            sql_r = 'UPDATE bankdata_copy SET cover_r =' + str(nodeRadius) + ',xq_house=' + str(total_house) + ',xzl_area=' + str(total_area) + ',near_node=' + str(r1['data']) + ',firmb_in_r=' + str(person_num) + ' WHERE name=' + "'" + x['name'] + "'"
            r4 = mysql.update(sql_r)
            mysql.dispose()



@route('/predict',method=['POST','OPTIONS'])
@allow_cross_domain 
def predict():
    if request.method == 'OPTIONS':
        return template('status:200')
    os.chdir("/home/czhou/python/model")
    gbr_r = joblib.load("pre_r_model.m")
    #户数 写字楼面积 周围网点数 银行 地区
    # test_X=[[150000,8300000,50,2,3]]
    print(request.json)
    loc = request.json['loc']
    region = request.json['region']
    bank = request.json['bank']
    test = [region,bank]
    test_r = [test]
    pre_r = gbr_r.predict(test_r)
    print(pre_r)

    mysql = Mysql()
    tmp = getNewLatLng(loc['lat'],loc['lng'],pre_r[0])
    netSql = 'SELECT COUNT(name) AS data from bankdata_copy WHERE lat >' + str(tmp['minLat']) + ' AND lat <' + str(tmp['maxLat']) + ' AND lng >' + str(tmp['minLng']) + ' AND lng <' + str(tmp['maxLng'])
    r1 = mysql.getOne(netSql)
    print(r1)
    firmSql = 'SELECT person_num from firm_info WHERE flat >' + str(tmp['minLat']) + ' AND flat <' + str(tmp['maxLat']) + ' AND flng >' + str(tmp['minLng']) + ' AND flng <' + str(tmp['maxLng'])
    r2 = mysql.getAll(firmSql)
    person_num = 0
    if r2 != False:
        for k in r2:
            person_num +=k['person_num']
    print(person_num)
    xqSql = 'SELECT total_house from fdd_xq WHERE lat >' + str(tmp['minLat']) + ' AND lat <' + str(tmp['maxLat']) + ' AND lng >' + str(tmp['minLng']) + ' AND lng <' + str(tmp['maxLng'])
    r3 = mysql.getAll(xqSql)
    total_house = 0
    if r3 != False:
        for k in r3:
            total_house +=k['total_house']
    print(total_house)
    mysql.dispose()
    
    test_b = [[r1['data'],person_num,total_house,region,bank]]
    gbr_b = joblib.load("pre_bsum_model.m")
    pre_b = gbr_b.predict(test_b)
    print(pre_b)
    return template('{"bsum":{{bsum}}}',bsum=pre_b)

@route('/dealPreR',method=['GET','POST'])
@allow_cross_domain 
def dealPreR():
    # 申请资源  
    mysql = Mysql()  
    sql = "delete FROM pre_r"  
    result = mysql.delete(sql)  
    #释放资源  
    mysql.dispose() 

    a = np.loadtxt('test_pre_r.csv',delimiter=',')
    a=np.delete(a,0,axis=0)
    for x in a :
        mysql = Mysql()
        tmpSql = "INSERT INTO pre_r (old_index, bsum, lat, lng, bank_index, region_index, pre_r) VALUES ("+str(x[0])+','+str(x[1])+','+str(x[2])+','+str(x[3])+','+str(x[5])+','+str(x[6])+','+str(x[7])+')'
        result = mysql.update(tmpSql)
        tmp = getNewLatLng(x[2],x[3],x[7])
        netSql = 'SELECT COUNT(name) AS data from bankdata_copy WHERE lat >' + str(tmp['minLat']) + ' AND lat <' + str(tmp['maxLat']) + ' AND lng >' + str(tmp['minLng']) + ' AND lng <' + str(tmp['maxLng'])
        r1 = mysql.getOne(netSql)
        print(r1)
        firmSql = 'SELECT person_num from firm_info WHERE flat >' + str(tmp['minLat']) + ' AND flat <' + str(tmp['maxLat']) + ' AND flng >' + str(tmp['minLng']) + ' AND flng <' + str(tmp['maxLng'])
        r2 = mysql.getAll(firmSql)
        person_num = 0
        for k in r2:
            person_num +=k['person_num']
        print(person_num)
        xqSql = 'SELECT total_house from fdd_xq WHERE lat >' + str(tmp['minLat']) + ' AND lat <' + str(tmp['maxLat']) + ' AND lng >' + str(tmp['minLng']) + ' AND lng <' + str(tmp['maxLng'])
        r3 = mysql.getAll(xqSql)
        total_house = 0
        for k in r3:
            total_house +=k['total_house']
        print(total_house)
        sql_r = 'UPDATE pre_r SET node_in_prer=' + str(r1['data']) + ',firm_in_prer=' + str(person_num) + ',house_in_prer=' + str(total_house) + ' WHERE old_index=' + str(x[0])
        r4 = mysql.update(sql_r)
        mysql.dispose()
    prebsum()
    return template('test:{{test}}',test="正在处理")

@route('/getscore',method=['GET','POST'])
@allow_cross_domain 
def getscore():
    # 申请资源  
    mysql = Mysql()  
    sql = "SELECT station_name FROM original_data WHERE station_name!='财务部' GROUP BY station_name"  
    nameList = mysql.getAll(sql)  
    #释放资源  
    mysql.dispose()
    loanM = []
    monthBsum = []
    replaceDeg = []
    bsumGrowth = []
    xqHouse = []
    xzlArea = []
    for x in nameList:
        mysql = Mysql()
        #每个网点的贷款金额
        loanOne = 0
        loanSql = 'SELECT loan_money FROM original_data WHERE station_name='+ "'" + x['station_name'] + "'"
        # loanRes = mysql.getAll(loanSql)
        # for k in loanRes:
        #     if k['loan_money']!=None :
        #         loanOne += k['loan_money']
        # loanM.append(loanOne)
        #单个网点每月业务总量
        bsumOne = 0
        bsumSql = 'SELECT count(_index) AS data FROM original_data WHERE station_name='+"'" + x['station_name'] + "'"
        # bsumRes = mysql.getOne(bsumSql)
        # if bsumRes['data']:
        #     monthBsum.append(bsumRes['data'])
        # else:
        #     monthBsum.append(bsumOne)
        #单个网点的可替代程度
        replaceSql = 'SELECT bratio,lg_to_min FROM bankdata_copy WHERE name='+"'" + x['station_name'] + "'"
        # replaceRes = mysql.getOne(replaceSql)
        # replaceDeg.append(replaceRes)
        #未来总业务量增长
        #网点定位 业务覆盖半径
        nodeLocSql = 'SELECT lng,lat,cover_r FROM bankdata_copy WHERE name='+"'" + x['station_name'] + "'"
        nodeLoc = mysql.getOne(nodeLocSql)
        tmp = getNewLatLng(nodeLoc['lat'],nodeLoc['lng'],nodeLoc['cover_r'])
        #网点周围新建小区 户数
        newXq = 0
        newXqSql = 'SELECT house_num from ajk_newxq WHERE lat >' + str(tmp['minLat']) + ' AND lat <' + str(tmp['maxLat']) + ' AND lng >' + str(tmp['minLng']) + ' AND lng <' + str(tmp['maxLng'])
        newXqRes = mysql.getAll(newXqSql)
        print(newXqRes)
        if newXqRes!=False:
            for k in newXqRes:
                if k['house_num']!=None:
                    newXq += k['house_num']
        xqHouse.append(newXq)
        #网点周围新建写字楼 面积
        newXzl = 0
        newXzlSql = 'SELECT area from ajk_newxzl WHERE lat >' + str(tmp['minLat']) + ' AND lat <' + str(tmp['maxLat']) + ' AND lng >' + str(tmp['minLng']) + ' AND lng <' + str(tmp['maxLng'])
        newXzlRes = mysql.getAll(newXzlSql)
        if newXzlRes!=False:
            for k in newXzlRes:
                newXzl += k['area']
        xzlArea.append(newXzl)
        mysql.dispose()
    sumxq = sum(xqHouse)
    sumxzl = sum(xzlArea)
    bsumGrowthScore = []
    for i,j in enumerate(xqHouse):
        tmp = j*0.06238087 + xzlArea[i]*0.16797319
        bsumGrowth.append(tmp)
    maxBG = max(bsumGrowth)
    mysql = Mysql()
    for i,j in enumerate(nameList):
        tmp = round((bsumGrowth[i]/maxBG)*100,2)
        tmpSql = 'UPDATE node_score SET future_bsum_growth=' + str(tmp) + ' WHERE _name='+ "'" +j['station_name'] + "'"
        mysql.update(tmpSql)
    mysql.dispose()

@route('/apriori',method=['POST','OPTIONS'])
@allow_cross_domain 
def apriori():
    if request.method == 'OPTIONS':
        return template('status:200')
    name = request.json['name']
    print(name)
    mysql = Mysql()
    taskSql = "SELECT * FROM trans_type_1"
    taskList = mysql.getAll(taskSql)
    nodeSql = "SELECT t_1,t_2,t_3,t_4,t_5,t_6,t_7 FROM task_seq WHERE station_name=" + "'" + name + "'"
    nodeTaskList  = mysql.getAll(nodeSql)
    mysql.dispose() 
    task = []
    for x in nodeTaskList:
        nodeTask = []
        for i in range(7):
            if x['t_'+str(i+1)] is not None:
                nodeTask.append(x['t_'+str(i+1)])
        task.append(nodeTask)
    df = pd.DataFrame({'task_seq':task})
    df.to_csv('task.csv',index=False,sep=',')
    inFile = dataFromFile('task.csv')
    items, rules = runApriori(inFile, 0.10, 0.5)
    rule = dealResults(taskList,rules)
    # return template('{"ret":{{ret}},"task":{{task}}}',task=rule,ret=200)
    return {"ret":200,"task":rule}

def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
        """calculates the support for items in the itemSet and returns a subset
       of the itemSet each of whose elements satisfies the minimum support"""
        _itemSet = set()
        localSet = defaultdict(int)

        for item in itemSet:
                for transaction in transactionList:
                        if item.issubset(transaction):
                                freqSet[item] += 1
                                localSet[item] += 1

        for item, count in localSet.items():
                support = float(count)/len(transactionList)

                if support >= minSupport:
                        _itemSet.add(item)

        return _itemSet


def joinSet(itemSet, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))              # Generate 1-itemSets
    return itemSet, transactionList


def runApriori(data_iter, minSupport, minConfidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    oneCSet = returnItemsWithMinSupport(itemSet,
                                        transactionList,
                                        minSupport,
                                        freqSet)

    currentLSet = oneCSet
    k = 2
    while(currentLSet != set([])):
        largeSet[k-1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(currentLSet,
                                                transactionList,
                                                minSupport,
                                                freqSet)
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
            """local function which Returns the support of an item"""
            return float(freqSet[item])/len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item))
                           for item in value])

    toRetRules = []
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item)/getSupport(element)
                    if confidence >= minConfidence:
                        toRetRules.append(((tuple(element), tuple(remain)),
                                           confidence))
    return toRetItems, toRetRules


def dealResults(task,rules):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    #for item, support in sorted(items, key=lambda item, support: support):
    tmp = []
    print(task[0]['_index'])
    for rule, confidence in sorted(rules, key=lambda rules: rules[1]):
        pre, post = rule
        pre = list(pre)
        post = list(post)
        for i,x in enumerate(pre):
            for y in task:
                if str(y['_index']) == str(x):
                    pre[i] = y['name'].strip()
        for i,x in enumerate(post):
            for y in task:
                if str(y['_index']) == str(x):
                    post[i] = y['name'].strip()
        tmp1 = {'tasked':pre,'tasking':post,'taskRatio':confidence}
        tmp.append(tmp1)
    print(tmp)
    return tmp

def dataFromFile(fname):
        """Function which reads from the file and yields a generator"""
        file_iter = open(fname, 'rU')
        for line in file_iter:
                line = onlyNum(line)
                # line = line.strip().rstrip(',')                        # Remove trailing comma
                record = frozenset(line.split(','))
                yield record

# 只保留字符串中的数字
def onlyNum(s,oth=''):
    s2 = s.lower();
    fomart = '0123456789,'
    for c in s2:
        if not c in fomart:
            s = s.replace(c,'');
    return s;

run(host='dev.engyne.net', port=8181) 
