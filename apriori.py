# -*- coding: utf-8 -*-
from bottle import *
import numpy as np
import pandas as pd
import json
from MySqlConn import Mysql  
from _sqlite3 import Row 
import math

import sys
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser


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
  
@route('/apriori',method=['POST','OPTIONS'])
@allow_cross_domain 
def apriori():
    name = request.json['name']
    print(name)
    mysql = Mysql('DB166')
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


run(host='localhost', port=8181) 