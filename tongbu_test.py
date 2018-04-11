import threading
import time
 
class Num:
    def __init__(self):
        self.num = 0
        self.lock = threading.Lock()
    def add(self):
        self.lock.acquire()#加锁，锁住相应的资源
        self.num += 1
        num = self.num
        self.lock.release()#解锁，离开该资源
        return num
 
n = Num()
class jdThread(threading.Thread):
    def __init__(self,item):
        threading.Thread.__init__(self)
        self.item = item
    def run(self):
        time.sleep(2)
        value = n.add()#将num加1，并输出原来的数据和+1之后的数据
        print(self.item,value)
 
for item in range(5):
    t = jdThread(item)
    t.start()
    t.join()#使线程一个一个执行