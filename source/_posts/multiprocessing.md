---
title: multiprocessing
date: 2021-08-20 14:50:29
tags:
categories:
password:
abstract:
message:
---







由于python的GIL锁的存在，导致在多线程的时候，同一时间只能有一个线程在cpu上运行，而且是单个cpu上运行，不管cpu有多少核数。如果要充分利用多核cpu的资源，在python中大部分情况下需要使用多进程。



# python多进程模块

Python中的多进程是通过multiprocessing包来实现的，和多线程的threading.Thread差不多，它可以利用multiprocessing.Process对象来创建一个进程对象。这个进程对象的方法和线程对象的方法差不多也有start(), run(), join()等方法，其中有一个方法不同Thread线程对象中的守护线程方法是setDeamon，而Process进程对象的守护进程是通过设置daemon属性来完成的。

<!--more-->

# 多进程的实现方法

1. 方法一

   ```
   from multiprocessing import  Process
   
   def fun1(name):
       print('测试%s多进程' %name)
   
   if __name__ == '__main__':
       process_list = []
       for i in range(5):  #开启5个子进程执行fun1函数
           p = Process(target=fun1,args=('Python',)) #实例化进程对象
           p.start()
           process_list.append(p)
   
       for i in process_list:
           p.join()
   
       print('结束测试')
   ```

   上面的代码开启了5个子进程去执行函数，我们可以观察结果，是同时打印的，这里实现了真正的并行操作，就是多个CPU同时执行任务。我们知道进程是python中最小的资源分配单元，也就是进程中间的数据，内存是不共享的，每启动一个进程，都要独立分配资源和拷贝访问的数据，所以进程的启动和销毁的代价是比较大了，所以在实际中使用多进程，要根据服务器的配置来设定。

2. 方法二

   ```
   from multiprocessing import  Process
   
   class MyProcess(Process): #继承Process类
       def __init__(self,name):
           super(MyProcess,self).__init__()
           self.name = name
   
       def run(self):
           print('测试%s多进程' % self.name)
   
   
   if __name__ == '__main__':
       process_list = []
       for i in range(5):  #开启5个子进程执行fun1函数
           p = MyProcess('Python') #实例化进程对象
           p.start()
           process_list.append(p)
   
       for i in process_list:
           p.join()
   
       print('结束测试')
   ```

   Process类的其他方法

   ```
   构造方法：
   
   Process([group [, target [, name [, args [, kwargs]]]]])
   　　group: 线程组 
   　　target: 要执行的方法
   　　name: 进程名
   　　args/kwargs: 要传入方法的参数
   
   实例方法：
   　　is_alive()：返回进程是否在运行,bool类型。
   　　join([timeout])：阻塞当前上下文环境的进程程，直到调用此方法的进程终止或到达指定的timeout（可选参数）。
   　　start()：进程准备就绪，等待CPU调度
   　　run()：strat()调用run方法，如果实例进程时未制定传入target，这star执行t默认run()方法。
   　　terminate()：不管任务是否完成，立即停止工作进程
   
   属性：
   　　daemon：和线程的setDeamon功能一样
   　　name：进程名字
   　　pid：进程号
   ```

   

# python 多进程的通信

进程是系统独立调度核分配系统资源（CPU、内存）的基本单位，进程之间是相互独立的，每启动一个新的进程相当于把数据进行了一次克隆，子进程里的数据修改无法影响到主进程中的数据，不同子进程之间的数据也不能共享，这是多进程在使用中与多线程最明显的区别。但是难道Python多进程中间难道就是孤立的吗？当然不是，python也提供了多种方法实现了多进程中间的通信和数据共享（可以修改一份数据）

### **进程对列Queue**

Queue在多线程中也说到过，在生成者消费者模式中使用，是线程安全的，是生产者和消费者中间的数据管道，那在python多进程中，它其实就是进程之间的数据管道，实现进程通信。

```
from multiprocessing import Process,Queue


def fun1(q,i):
    print('子进程%s 开始put数据' %i)
    q.put('我是%s 通过Queue通信' %i)

if __name__ == '__main__':
    q = Queue()

    process_list = []
    for i in range(3):
        p = Process(target=fun1,args=(q,i,))  #注意args里面要把q对象传给我们要执行的方法，这样子进程才能和主进程用Queue来通信
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()

    print('主进程获取Queue数据')
    print(q.get())
    print(q.get())
    print(q.get())
    print('结束测试')
```



### **管道Pipe**

管道Pipe和Queue的作用大致差不多，也是实现进程间的通信，下面之间看怎么使用吧

```
from multiprocessing import Process, Pipe
def fun1(conn):
    print('子进程发送消息：')
    conn.send('你好主进程')
    print('子进程接受消息：')
    print(conn.recv())
    conn.close()

if __name__ == '__main__':
    conn1, conn2 = Pipe() #关键点，pipe实例化生成一个双向管
    p = Process(target=fun1, args=(conn2,)) #conn2传给子进程
    p.start()
    print('主进程接受消息：')
    print(conn1.recv())
    print('主进程发送消息：')
    conn1.send("你好子进程")
    p.join()
    print('结束测试')
```



```
## 进程池不能使用Queue,而要用Pipe, 但是可以使用Manager包装一下
from multiprocessing import Pool, Process,Queue, Pipe
import time,random

def consumer(q, name):
    # q, name = k
    while True:
        food = q.recv()
        if food is None:
            print('接收到了一个空,生产者已经完事了')
            q.close()
            break

        print('\033[31m{}消费了{}\033[0m'.format(name,food))
        time.sleep(random.random())

def producer(name,food, q):
    # print(name, food)
    # name,food,q = k
    for i in range(10):
        time.sleep(random.random())
        f = '{}生产了{}{}'.format(name,food,i)
        print(f)
        q.send(f)
    q.send(None)


if __name__ == '__main__':
    start_time = time.time()
    q1,q2 = Pipe()

    p1 = Pool(3)
    p2 = Pool(3)

    # p1 = Process(target=producer,args=('fioman','包子',q))
    # p2 = Process(target=producer,args=('jingjing','馒头',q))
    # p1.start()
    # p2.start()
    # for i in [('fioman','包子',q), ('jingjing','馒头',q)]:
    #     print(i)
        # p1.apply_async(func=producer, args=(i,))
    p1.apply_async(func=producer, args=('fioman','包子',q1))
    p1.apply_async(func=producer, args=('jingjing','馒头',q1))
    # p1.apply_async(func=producer, args=('hu','馒头',q1))

    # for i in [(q,'mengmeng'), (q,'xiaoxiao')]:
    p2.apply_async(func=consumer, args=(q2,'mengmeng'))
    p2.apply_async(func=consumer, args=(q2,'xiaoxiao'))
    p2.apply_async(func=consumer, args=(q2,'x'))
    p2.apply_async(func=consumer, args=(q2,'xy'))




    # c1 = Process(target=consumer,args=(q,'mengmeng'))
    # c2 = Process(target=consumer,args=(q,'xiaoxiao'))
    p1.close()
    p2.close()
    p1.join()
    # c1.start()
    # c2.start()

    # 让主程序可以等待子进程的结束.
    # p1.join()
    # p2.join()
    # 生产者的进程结束,这里需要放置两个空值,供消费者获取,用来判断已经没有存货了
    # q.put(None)
    # q.put(None)

    print('主程序结束..........')
    end_time = time.time()
    print(end_time - start_time)
```





### **Managers**

Queue和Pipe只是实现了数据交互，并没实现数据共享，即一个进程去更改另一个进程的数据。那么就要用到Managers

```
from multiprocessing import Process, Manager

def fun1(dic,lis,index):

    dic[index] = 'a'
    dic['2'] = 'b'    
    lis.append(index)    #[0,1,2,3,4,0,1,2,3,4,5,6,7,8,9]
    #print(l)

if __name__ == '__main__':
    with Manager() as manager:
        dic = manager.dict()#注意字典的声明方式，不能直接通过{}来定义
        l = manager.list(range(5))#[0,1,2,3,4]

        process_list = []
        for i in range(10):
            p = Process(target=fun1, args=(dic,l,i))
            p.start()
            process_list.append(p)

        for res in process_list:
            res.join()
        print(dic)
        print(l)
```

# 进程池

进程池内部维护一个进程序列，当使用时，则去进程池中获取一个进程，如果进程池序列中没有可供使用的进程，那么程序就会等待，直到进程池中有可用进程为止。就是固定有几个进程可以使用。

进程池中有两个方法：

apply：同步，一般不使用

apply_async：异步

```
from  multiprocessing import Process,Pool
import os, time, random

def fun1(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    pool = Pool(5) #创建一个5个进程的进程池

    for i in range(10):
        pool.apply_async(func=fun1, args=(i,))

    pool.close()
    pool.join()
    print('结束测试')
```

对`Pool`对象调用`join()`方法会等待所有子进程执行完毕，调用`join()`之前必须先调用`close()`，调用`close()`之后就不能继续添加新的`Process`了。

### 	**进程池map方法** 

```
import os 
import PIL 

from multiprocessing import Pool 
from PIL import Image

SIZE = (75,75)
SAVE_DIRECTORY = \'thumbs\'

def get_image_paths(folder):
    return (os.path.join(folder, f) 
            for f in os.listdir(folder) 
            if \'jpeg\' in f)

def create_thumbnail(filename): 
    im = Image.open(filename)
    im.thumbnail(SIZE, Image.ANTIALIAS)
    base, fname = os.path.split(filename) 
    save_path = os.path.join(base, SAVE_DIRECTORY, fname)
    im.save(save_path)

if __name__ == \'__main__\':
    folder = os.path.abspath(
        \'11_18_2013_R000_IQM_Big_Sur_Mon__e10d1958e7b766c3e840\')
    os.mkdir(os.path.join(folder, SAVE_DIRECTORY))

    images = get_image_paths(folder)

    pool = Pool()
    pool.map(creat_thumbnail, images) #关键点，images是一个可迭代对象
    pool.close()
    pool.join()
```

## map map_async, apply apply_async

```

import multiprocessing
import time
 
 
def func(msg):
    print('msg: ', msg)
    time.sleep(1)
    print('********')
    return 'func_return: %s' % msg
 
if __name__ == '__main__':
    # apply_async
    print('\n--------apply_async------------')
    pool = multiprocessing.Pool(processes=4)
    results = []
    for i in range(10):
        msg = 'hello world %d' % i
        result = pool.apply_async(func, (msg, ))
        results.append(result)
    print('apply_async: 不堵塞')
 
    for i in results:
        i.wait()  # 等待进程函数执行完毕
 
    for i in results:
        if i.ready():  # 进程函数是否已经启动了
            if i.successful():  # 进程函数是否执行成功
                print(i.get())  # 进程函数返回值
 
    # apply
    print('\n--------apply------------')
    pool = multiprocessing.Pool(processes=4)
    results = []
    for i in range(10):
        msg = 'hello world %d' % i
        result = pool.apply(func, (msg,))
        results.append(result)
    print('apply: 堵塞')  # 执行完func才执行该句
    pool.close()
    pool.join()  # join语句要放在close之后
    print(results)
 
    # map
    print('\n--------map------------')
    args = [1, 2, 4, 5, 7, 8]
    pool = multiprocessing.Pool(processes=5)
    return_data = pool.map(func, args)
    print('堵塞')  # 执行完func才执行该句
    pool.close()
    pool.join()  # join语句要放在close之后
    print(return_data)
 
    # map_async
    print('\n--------map_async------------')
    pool = multiprocessing.Pool(processes=5)
    result = pool.map_async(func, args)
    print('ready: ', result.ready())
    print('不堵塞')
    result.wait()  # 等待所有进程函数执行完毕
 
    if result.ready():  # 进程函数是否已经启动了
        if result.successful():  # 进程函数是否执行成功
            print(result.get())  # 进程函数返回值
```





# 参考方法

[一篇文章搞定Python多进程](https://zhuanlan.zhihu.com/p/64702600)

[python 如何优雅地退出子进程](https://blog.csdn.net/ubuntu64fan/article/details/51898740)

[ python多进程多线程,多个程序同时运行](https://blog.csdn.net/qq_43475705/article/details/115518463)

[pytorch多模型异步推理](https://blog.csdn.net/qq_17127427/article/details/116532097)

[python进程池multiprocessing.Pool和线程池multiprocessing.dummy.Pool实例 - dylan9 - 博客园 (cnblogs.com)](https://www.cnblogs.com/dylan9/p/9207366.html)
