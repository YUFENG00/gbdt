# _*_ encoding utf-8 _*_
import numpy as np

if __name__ == '__main__':
    a = np.loadtxt('bhan.csv', delimiter=',')
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
    df = np.zeros((n, m))
    for i in range(n):
        df[i, :] = list_b[i]
    print(b)