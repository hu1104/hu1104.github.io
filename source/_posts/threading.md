---
title: threading
date: 2021-08-20 15:37:55
tags:
categories:
password:
abstract:
message:
---





# python 多线程实现

首先，python的多线程是假的。

看一个例子来看看python多线程的实现。

```
import threading
import time

def say(name):
        print('你好%s at %s' %(name,time.ctime()))
        time.sleep(2)
        print("结束%s at %s" %(name,time.ctime()))

def listen(name):
    print('你好%s at %s' % (name,time.ctime()))
    time.sleep(4)
    print("结束%s at %s" % (name,time.ctime()))

if __name__ == '__main__':
    t1 = threading.Thread(target=say,args=('tony',))  #创建线程对象，Thread是一个类，实例化产生t1对象，这里就是创建了一个线程对象t1
    t1.start() #启动线程，线程执行
    t2 = threading.Thread(target=listen, args=('simon',)) #这里就是创建了一个线程对象t2
    t2.start()

    print("程序结束=====================")
# 输出结果为
你好tony at Fri Aug 20 15:42:07 2021 --t1线程执行
你好simon at Fri Aug 20 15:42:07 2021 --t2线程执行
程序结束===================== --主线程执行
结束tony at Fri Aug 20 15:42:09 2021 --sleep之后，t1线程执行
结束simon at Fri Aug 20 15:42:11 2021 --sleep之后，t2线程执行
```

我们可以看到主线程的print并不是等t1,t2线程都执行完毕之后才打印的，这是因为主线程和t1,t2 线程是同时跑的。但是主进程要等非守护子线程结束之后，主线程才会退出。

<!--more-->

上面其实就是python多线程的最简单用法，但是，一般开发中，我们需要主线程的print打印是在最后面的，表明所有流程都结束了，也就是主线程结束了。这里就引入了一个join的概念。

```
import threading
import time

def say(name):
        print('你好%s at %s' %(name,time.ctime()))
        time.sleep(2)
        print("结束%s at %s" %(name,time.ctime()))

def listen(name):
    print('你好%s at %s' % (name,time.ctime()))
    time.sleep(4)
    print("结束%s at %s" % (name,time.ctime()))

if __name__ == '__main__':
    t1 = threading.Thread(target=say,args=('tony',))  #Thread是一个类，实例化产生t1对象，这里就是创建了一个线程对象t1
    t1.start() #线程执行
    t2 = threading.Thread(target=listen, args=('simon',)) #这里就是创建了一个线程对象t2
    t2.start()

    t1.join() #join等t1子线程结束，主线程打印并且结束
    t2.join() #join等t2子线程结束，主线程打印并且结束
    print("程序结束=====================")
#输出
你好tony at Fri Aug 20 15:49:01 2021
你好simon at Fri Aug 20 15:49:01 2021
结束tony at Fri Aug 20 15:49:03 2021
结束simon at Fri Aug 20 15:49:05 2021
程序结束=====================
```

上面代码中加入join方法后实现了，我们上面所想要的结果，主线程print最后执行，并且主线程退出，注意主线程执行了打印操作和主线程结束不是一个概念，如果子线程不加join，则主线程也会执行打印，但是主线程不会结束，还是需要待非守护子线程结束之后，主线程才结束。



上面的情况，主进程都需要等待非守护子线程结束之后，主线程才结束。那我们是不是注意到一点，我说的是“非守护子线程”，那什么是非守护子线程？默认的子线程都是主线程的非守护子线程，但是有时候我们有需求，当主进程结束，不管子线程有没有结束，子线程都要跟随主线程一起退出，这时候我们引入一个“守护线程”的概念。



如果某个子线程设置为守护线程，主线程其实就不用管这个子线程了，当所有其他非守护线程结束，主线程就会退出，而守护线程将和主线程一起退出，守护主线程，这就是守护线程的意思



1. 设置t1线程为守护线程，看看执行结果

   ```
   import threading
   import time
   
   def say(name):
           print('你好%s at %s' %(name,time.ctime()))
           time.sleep(2)
           print("结束%s at %s" %(name,time.ctime()))
   
   def listen(name):
       print('你好%s at %s' % (name,time.ctime()))
       time.sleep(4)
       print("结束%s at %s" % (name,time.ctime()))
   
   if __name__ == '__main__':
       t1 = threading.Thread(target=say,args=('tony',))  #Thread是一个类，实例化产生t1对象，这里就是创建了一个线程对象t1
       t1.setDaemon(True)
       t1.start() #线程执行
       t2 = threading.Thread(target=listen, args=('simon',)) #这里就是创建了一个线程对象t2
       t2.start()
   
       print("程序结束=====================")
   #输出
   你好tony at Fri Aug 20 15:59:52 2021
   你好simon at Fri Aug 20 15:59:52 2021程序结束=====================
   
   结束tony at Fri Aug 20 15:59:54 2021
   结束simon at Fri Aug 20 15:59:56 2021
   ```

   

2. 设置t2线程为守护线程，看看执行结果

   ```
   import threading
   import time
   
   def say(name):
           print('你好%s at %s' %(name,time.ctime()))
           time.sleep(2)
           print("结束%s at %s" %(name,time.ctime()))
   
   def listen(name):
       print('你好%s at %s' % (name,time.ctime()))
       time.sleep(4)
       print("结束%s at %s" % (name,time.ctime()))
   
   if __name__ == '__main__':
       t1 = threading.Thread(target=say,args=('tony',))  #Thread是一个类，实例化产生t1对象，这里就是创建了一个线程对象t1
       
       t1.start() #线程执行
       t2 = threading.Thread(target=listen, args=('simon',)) #这里就是创建了一个线程对象t2
       t2.setDaemon(True)
       t2.start()
   
       print("程序结束=====================")
   #输出
   你好tony at Fri Aug 20 16:02:19 2021
   你好simon at Fri Aug 20 16:02:19 2021程序结束=====================
   
   结束tony at Fri Aug 20 16:02:21 2021
   ```

   

不知道大家有没有弄清楚上面python多线程的实现方式以及join,守护线程的用法。

多线程的实现方法：

1. 直接创建子进程

   首先可以使用 Thread 类来创建一个线程，创建时需要指定 target 参数为运行的方法名称，如果被调用的方法需要传入额外的参数，则可以通过 Thread 的 args 参数来指定。

   ```
   import threading
   import time
   def target(second):
       print(f'Threading {threading.current_thread().name} is running')
       print(f'Threading {threading.current_thread().name} sleep {second}s')
       time.sleep(second)
       print(f'Threading {threading.current_thread().name} is ended')
   print(f'Threading {threading.current_thread().name} is running')
   for i in [1, 5]:
       thread = threading.Thread(target=target, args=[i])
       thread.start()
   ```

   

2. 继承Thread类创建子进程

   通过继承 Thread 类的方式创建一个线程，该线程需要执行的方法写在类的 run 方法里面即可。

   ```
   import threading
   import time
   class MyThread(threading.Thread):
       def __init__(self, second):
           threading.Thread.__init__(self)
           self.second = second
       
       def run(self):
           print(f'Threading {threading.current_thread().name} is running')
           print(f'Threading {threading.current_thread().name} sleep {self.second}s')
           time.sleep(self.second)
           print(f'Threading {threading.current_thread().name} is ended')
   print(f'Threading {threading.current_thread().name} is running')
   threads = []
   for i in [1, 5]:
       thread = MyThread(i)
       threads.append(thread)
       thread.start()
   ```

   

主要方法：



join()：在子线程完成运行之前，这个子线程的父线程将一直被阻塞。

setDaemon(True)：

将线程声明为守护线程，必须在start() 方法调用之前设置， 如果不设置为守护线程程序会被无限挂起。这个方法基本和join是相反的。

当我们在程序运行中，执行一个主线程，如果主线程又创建一个子线程，主线程和子线程 就分兵两路，分别运行，那么当主线程完成

想退出时，会检验子线程是否完成。如 果子线程未完成，则主线程会等待子线程完成后再退出。但是有时候我们需要的是 只要主线程完成了，不管子线程是否完成，都要和主线程一起退出，这时就可以 用setDaemon方法啦。



其他方法：



run(): 线程被cpu调度后自动执行线程对象的run方法
start():启动线程活动。
isAlive(): 返回线程是否活动的。
getName(): 返回线程名。
setName(): 设置线程名。



threading模块提供的一些方法：
threading.currentThread(): 返回当前的线程变量。
threading.enumerate(): 返回一个包含正在运行的线程的list。正在运行指线程启动后、结束前，不包括启动前和终止后的线程。
threading.activeCount():返回正在运行的线程数量，与len(threading.enumerate())有相同的结果。





上面的例子中我们注意到两如果个任务如果顺序执行要6s结束，如果是多线程执行4S结束，性能是有所提升的，但是我们要知道这里的性能提升实际上是由于cpu并发实现性能提升，也就是cpu线程切换（多道技术）带来的，而并不是真正的多cpu并行执行。



上面提到了并行和并发，那这两者有什么区别呢？

并发：是指一个系统具有处理多个任务的能力（cpu切换，多道技术）
并行：是指一个系统具有同时处理多个任务的能力（cpu同时处理多个任务）
并行是并发的一种情况，子集



# python同步锁

锁通常被用来实现对共享资源的同步访问。为每一个共享资源创建一个Lock对象，当你需要访问该资源时，调用acquire方法来获取锁对象（如果其它线程已经获得了该锁，则当前线程需等待其被释放），待资源访问完后，再调用release方法释放锁。

 当没有同步锁时：

```
import threading
import time

num = 100

def fun_sub():
    global num
    # num -= 1
    num2 = num
    time.sleep(0.001)
    num = num2-1

if __name__ == '__main__':
    print('开始测试同步锁 at %s' % time.ctime())

    thread_list = []
    for thread in range(100):
        t = threading.Thread(target=fun_sub)
        t.start()
        thread_list.append(t)

    for t in thread_list:
        t.join()
    print('num is %d' % num)
    print('结束测试同步锁 at %s' % time.ctime())
# 输出
开始测试同步锁 at Fri Aug 20 17:45:40 2021
num is 98
结束测试同步锁 at Fri Aug 20 17:45:40 2021
```

上面的例子其实很简单就是创建100的线程，然后每个线程去从公共资源num变量去执行减1操作，按照正常情况下面，等到代码执行结束，打印num变量，应该得到的是0，因为100个线程都去执行了一次减1的操作。但我们会发现，每次执行的结果num值都不是一样的。



我们来看看上面代码的执行流程。
1.因为GIL，只有一个线程（假设线程1）拿到了num这个资源，然后把变量赋值给num2,sleep 0.001秒，这时候num=100
2.当第一个线程sleep 0.001秒这个期间，这个线程会做yield操作，就是把cpu切换给别的线程执行（假设线程2拿到个GIL，获得cpu使用权），线程2也和线程1一样也拿到num,返回赋值给num2，然sleep,这时候，其实num还是=100.
3.线程2 sleep时候，又要yield操作，假设线程3拿到num,执行上面的操作，其实num有可能还是100
4.等到后面cpu重新切换给线程1，线程2，线程3上执行的时候，他们执行减1操作后，其实等到的num其实都是99，而不是顺序递减的。
5.其他剩余的线程操作如上



加上锁后：

```
import threading
import time

num = 100

def fun_sub():
    global num
    lock.acquire()
    print('----加锁----')
    print('现在操作共享资源的线程名字是:',t.name)
    num2 = num
    time.sleep(0.001)
    num = num2-1
    lock.release()
    print('----释放锁----')

if __name__ == '__main__':
    print('开始测试同步锁 at %s' % time.ctime())

    lock = threading.Lock() #创建一把同步锁

    thread_list = []
    for thread in range(100):
        t = threading.Thread(target=fun_sub)
        t.start()
        thread_list.append(t)

    for t in thread_list:
        t.join()
    print('num is %d' % num)
    print('结束测试同步锁 at %s' % time.ctime())
```



## 死锁

死锁的这个概念在很多地方都存在，比较在数据中，大概介绍下私有是怎么产生的

1. A拿了一个苹果
2. B拿了一个香蕉
3. A现在想再拿个香蕉，就在等待B释放这个香蕉
4. B同时想要再拿个苹果，这时候就等待A释放苹果
5. 这样就是陷入了僵局，这就是生活中的死锁



python中在线程间共享多个资源的时候，如果两个线程分别占有一部分资源并且同时等待对方的资源，就会造成死锁，因为系统判断这部分资源都正在使用，所有这两个线程在无外力作用下将一直等待下去。下面是一个死锁的例子：

```
import threading
import time

lock_apple = threading.Lock()
lock_banana = threading.Lock()

class MyThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        self.fun1()
        self.fun2()

    def fun1(self):

        lock_apple.acquire()  # 如果锁被占用,则阻塞在这里,等待锁的释放

        print ("线程 %s , 想拿: %s--%s" %(self.name, "苹果",time.ctime()))

        lock_banana.acquire()
        print ("线程 %s , 想拿: %s--%s" %(self.name, "香蕉",time.ctime()))
        lock_banana.release()
        lock_apple.release()


    def fun2(self):

        lock_banana.acquire()
        print ("线程 %s , 想拿: %s--%s" %(self.name, "香蕉",time.ctime()))
        time.sleep(0.1)

        lock_apple.acquire()
        print ("线程 %s , 想拿: %s--%s" %(self.name, "苹果",time.ctime()))
        lock_apple.release()

        lock_banana.release()

if __name__ == "__main__":
    for i in range(0, 10):  #建立10个线程
        my_thread = MyThread()  #类继承法是python多线程的另外一种实现方式
        my_thread.start()
```

上面的代码其实就是描述了苹果和香蕉的故事。大家可以仔细看看过程。下面我们看看执行流程

1.fun1中，线程1先拿了苹果，然后拿了香蕉，然后释放香蕉和苹果，然后再在fun2中又拿了香蕉，sleep 0.1秒。
2.在线程1的执行过程中，线程2进入了，因为苹果被线程1释放了，线程2这时候获得了苹果，然后想拿香蕉
3.这时候就出现问题了，线程一拿完香蕉之后想拿苹果，返现苹果被线程2拿到了，线程2拿到苹果执行，想拿香蕉，发现香蕉被线程1持有了
4.双向等待，出现死锁，代码执行不下去了



## Python递归锁RLock

为了支持在同一线程中多次请求同一资源，python提供了"递归锁"：threading.RLock。RLock内部维护着一个Lock和一个counter变量，counter记录了acquire的次数，从而使得资源可以被多次acquire。直到一个线程所有的acquire都被release，其他的线程才能获得资源。

下面我们用递归锁RLock解决上面的死锁问题:

```
import threading
import time

lock = threading.RLock()  #递归锁


class MyThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        self.fun1()
        self.fun2()

    def fun1(self):

        lock.acquire()  # 如果锁被占用,则阻塞在这里,等待锁的释放

        print ("线程 %s , 想拿: %s--%s" %(self.name, "苹果",time.ctime()))

        lock.acquire()
        print ("线程 %s , 想拿: %s--%s" %(self.name, "香蕉",time.ctime()))
        lock.release()
        lock.release()


    def fun2(self):

        lock.acquire()
        print ("线程 %s , 想拿: %s--%s" %(self.name, "香蕉",time.ctime()))
        time.sleep(0.1)

        lock.acquire()
        print ("线程 %s , 想拿: %s--%s" %(self.name, "苹果",time.ctime()))
        lock.release()

        lock.release()

if __name__ == "__main__":
    for i in range(0, 10):  #建立10个线程
        my_thread = MyThread()  #类继承法是python多线程的另外一种实现方式
        my_thread.start()
```

上面我们用一把递归锁，就解决了多个同步锁导致的死锁问题。大家可以把RLock理解为大锁中还有小锁，只有等到内部所有的小锁，都没有了，其他的线程才能进入这个公共资源。

另外一点前面没有就算用类继承的方法实现python多线程，这个大家可以查下，就算继承Thread类，然后重新run方法来实现。



# 同步条件

先说说为什么我们需要这个同步条件，我们的python多线程在执行task过程中，是相互竞争的，大家都可以先获取cpu的执行权限，这就是问题所在的地方，每个线程都是独立运行且状态不可预测，但是我们想想如果我们的业务中需要根据情况来决定线程的执行顺序，也就是程序中的其他线程需要通过判断某个线程的状态来确定自己下一步的操作，这时候我们就需要使用threading库中的Event对象。 对象包含一个可由线程设置的信号标志,它允许线程等待某些事件的发生。



在 初始情况下,Event对象中的信号标志被设置为假，如果有线程等待一个Event对象, ,那么这个线程将会被一直阻塞直至该标志为真。



一个线程如果将一个Event对象的信号标志设置为真,它将唤醒所有等待这个Event对象的线程继续执行。



Event的方法如下：

```
event.isSet()：返回event的状态值
event.wait()：如果 event.isSet()==False，将阻塞线程触发event.wait()
event.set()： 设置event的状态值为True，所有阻塞池的线程激活进入就绪状态， 等待执行
event.clear()：恢复event的状态值为False
```

```
import threading
import time

class Teacher(threading.Thread):
    def run(self):
        print("大家现在要考试")
        print(event.isSet())
        event.set()
        time.sleep(3)
        print("考试结束")
        print(event.isSet())
        event.set()
class Student(threading.Thread):
    def run(self):
        event.wait()
        print("啊啊啊啊啊啊")
        time.sleep(1)
        event.clear()
        event.wait()
        print("下课回家")

if __name__=="__main__":
    event=threading.Event()
    threads=[]
    for i in range(10):
        threads.append(Student())
    threads.append(Teacher())
    for t in threads:
        t.start()
    for t in threads:
        t.join()
```



上述代码流程

```
1.模拟1个老师和10个学生，进行考试，我们需要的目的是学生线程要等待老师线程说完“大家现在考试”，然后学生线程去考试，之后老师线程说“考试结束”，学生线程放学回家，学生线程的执行与否取决于老师线程，所以这里用的Event
2.学生线程开始event.wait()，这个说明如果event如果一直不设置的话，学生线程就一直等待，等待一个event.set()操作
3.老师线程说完"大家现在要考试"，然后event.set()，执行event,设置完执行，学生线程就能够被唤醒继续执行下面的操作发出"啊啊啊啊啊啊"的叫苦连天
4.学生线程进行考试，并且执行event.clear()，清除event，因为他们在等老师说“考试结束”，之后他们在等老师线程的event.set()
5.老师线程执行event.set()，唤醒学生线程，然后下课回家.
```



## 信号量

信号量用来控制线程并发数的，Semaphore管理一个内置的计数 器，每当调用acquire()时-1，调用release()时+1。计数器不能小于0，当计数器为 0时，acquire()将阻塞线程至同步锁定状态，直到其他线程调用release()。其实就是控制最多几个线程可以操作同享资源。



```
import threading
import time

semaphore = threading.Semaphore(5)

def func():
    if semaphore.acquire():
        print (threading.currentThread().getName() + '获取共享资源')
        time.sleep(2)
        semaphore.release()

for i in range(10)
  t1 = threading.Thread(target=func)
  t1.start()
```

上面一个简单的例子就是创建10个线程，让每次只让5个线程去执行func函数。



## 队列

Queue是python标准库中的线程安全的队列实现,提供了一个适用于多线程编程的先进先出的数据结构，即队列，用来在生产者和消费者线程之间的信息传递



```
import threading,time

m=[1,2,3,4,5]
print(m[-1])

def remove_last():
    a=m[-1]
    time.sleep(1)
    m.remove(a)


t1=threading.Thread(target=remove_last)
t1.start()

t2=threading.Thread(target=remove_last)
t2.start()
## list 不是线程安全会报错
```



```
创建一个“队列”对象
import Queue
q = Queue.Queue(maxsize = 10)
Queue.Queue类即是一个队列的同步实现。队列长度可为无限或者有限。可通过Queue的构造函数的可选参数maxsize来设定队列长度。如果maxsize小于1就表示队列长度无限。

将一个值放入队列中
q.put(10)
调用队列对象的put()方法在队尾插入一个项目。put()有两个参数，第一个item为必需的，为插入项目的值；第二个block为可选参数，默认为
1。如果队列当前为空且block为1，put()方法就使调用线程暂停,直到空出一个数据单元。如果block为0，put方法将引发Full异常。

将一个值从队列中取出
q.get()
调用队列对象的get()方法从队头删除并返回一个项目。可选参数为block，默认为True。如果队列为空且block为True，
get()就使调用线程暂停，直至有项目可用。如果队列为空且block为False，队列将引发Empty异常。

Python Queue模块有三种队列及构造函数:
1、Python Queue模块的FIFO队列先进先出。   class queue.Queue(maxsize)
2、LIFO类似于堆，即先进后出。               class queue.LifoQueue(maxsize)
3、还有一种是优先级队列级别越低越先出来。        class queue.PriorityQueue(maxsize)

此包中的常用方法(q = Queue.Queue()):
q.qsize() 返回队列的大小
q.empty() 如果队列为空，返回True,反之False
q.full() 如果队列满了，返回True,反之False
q.full 与 maxsize 大小对应
q.get([block[, timeout]]) 获取队列，timeout等待时间
q.get_nowait() 相当q.get(False)
非阻塞 q.put(item) 写入队列，timeout等待时间
q.put_nowait(item) 相当q.put(item, False)
q.task_done() 在完成一项工作之后，q.task_done() 函数向任务已经完成的队列发送一个信号
q.join() 实际上意味着等到队列为空，再执行别的操作
```

队列(queue)一般会被用在生产者和消费者模型上。

生产者消费者模型：

为什么要使用生产者和消费者模式

在python线程中，生产者就是生产数据的线程，消费者就是消费数据的线程。在多线程开发当中，如果生产者处理速度很快，而消费者处理速度很慢，那么生产者就必须等待消费者处理完，才能继续生产数据。同样的道理，如果消费者的处理能力大于生产者，那么消费者就必须等待生产者。为了解决这个问题于是引入了生产者和消费者模式。

什么是生产者消费者模式

生产者消费者模式是通过一个容器来解决生产者和消费者的强耦合问题。生产者和消费者彼此之间不直接通讯，而通过阻塞队列来进行通讯，所以生产者生产完数据之后不用等待消费者处理，直接扔给阻塞队列，消费者不找生产者要数据，而是直接从阻塞队列里取，阻塞队列就相当于一个缓冲区，平衡了生产者和消费者的处理能力。

下面我们看看生产者消费者的代码，就拿大家常说的吃包子为例子吧



```
import time,random
import queue,threading

q = queue.Queue()

def Producer(name):
  count = 0
  while count <10:
    print("制造包子ing")
    time.sleep(random.randrange(3))
    q.put(count)
    print('生产者 %s 生产了 %s 包子..' %(name, count))
    count +=1
    #q.task_done()
    #q.join()

def Consumer(name):
  count = 0
  while count <10:
    time.sleep(random.randrange(4))
    if not q.empty():
        data = q.get()
        #q.task_done()
        #q.join()
        print(data)
        print('消费者 %s 消费了 %s 包子...' %(name, data))
    else:
        print("包子吃完了")
    count +=1

c1 = threading.Thread(target=Producer, args=('小明',))
c2 = threading.Thread(target=Consumer, args=('小花',))
c3 = threading.Thread(target=Consumer, args=('小灰',))
c1.start()
c2.start()
c3.start()

c1.join()
c2.join()
c3.join()

print('结束')
```

另一种实现方式

```
# q.task_done() 在完成一项工作之后，q.task_done() 函数向任务已经完成的队列发送一个信号
# q.join() 实际上意味着等到队列为空，再执行别的操作
import time,random
import queue,threading

q = queue.Queue()

def Producer(name):
  count = 0
  while count <10:
    print("制造包子ing")
    time.sleep(random.randrange(3))
    q.put(count)
    print('生产者 %s 生产了 %s 包子..' %(name, count))
    count +=1
    q.task_done()
    #q.join()

def Consumer(name):
  count = 0
  while count <10:
    time.sleep(random.randrange(4))
    data = q.get()
    #q.task_done()
    print('等待中')
    q.join()
    print('消费者 %s 消费了 %s 包子...' %(name, data))
    count +=1

c1 = threading.Thread(target=Producer, args=('小明',))
c2 = threading.Thread(target=Consumer, args=('小花',))
c3 = threading.Thread(target=Consumer, args=('小灰',))
c4 = threading.Thread(target=Consumer, args=('小天',))

c1.start()
c2.start()
c3.start()
c4.start()
```



# 参考资料

[一篇文章搞懂Python多线程简单实现和GIL](https://mp.weixin.qq.com/s/Hgp-x-T3ss4IiVk2_4VUrA)

[一篇文章理清Python多线程同步锁，死锁和递归锁](https://mp.weixin.qq.com/s/RZSBe2MG9tsbUVZLHxK9NA)

[同步条件](https://mp.weixin.qq.com/s/vKsNbDZnvg6LHWVA-AOIMA)





















