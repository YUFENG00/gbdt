#coding:utf-8  
''''' 
 
@author: yufeng 
'''  
from MySqlConn import Mysql  
from _sqlite3 import Row  
  
#申请资源  
mysql = Mysql()  
  
sqlAll = "SELECT _index,bankName,region,lable FROM bankdata WHERE _index < 10"  
result = mysql.getAll(sqlAll)  
if result :    
    for row in result :  
        print("%s\t%s"%(row["_index"],row["bankName"]))  
sqlAll = "SELECT _index,bankName,region,lable FROM bankdata WHERE _index < 10"  
result = mysql.getMany(sqlAll,2)  
if result :  
    for row in result :  
        print("%s\t%s"%(row["_index"],row["bankName"]))          
          
          
result = mysql.getOne(sqlAll)   
print("%s\t%s"%(result["_index"],result["bankName"]))  
  
#释放资源  
mysql.dispose() 