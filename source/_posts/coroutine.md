---
title: coroutine
date: 2021-08-24 16:49:44
tags:
categories:
password:
abstract:
message:
---

### asyncio + yield from

```
#-*- coding:utf8 -*-
import asyncio

@asyncio.coroutine
def test(i):
    print('test_1', i)
    r = yield from asyncio.sleep(1)
    print('test_2', i)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    tasks = [test(i) for i in range(3)]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
```

`@asyncio.coroutine`把一个generator标记为coroutine类型，然后就把这个coroutine扔到EventLoop中执行。test()会首先打印出test_1，然后`yield from`语法可以让我们方便地调用另一个generator。由于`asyncio.sleep()`也是一个coroutine，所以线程不会等待`asyncio.sleep()`，而是直接中断并执行下一个消息循环。当`asyncio.sleep()`返回时，线程就可以从`yield from`拿到返回值（此处是None），然后接着执行下一行语句。把`asyncio.sleep(1)`看成是一个耗时1秒的IO操作，在此期间主线程并未等待，而是去执行EventLoop中其他可以执行的coroutine了，因此可以实现并发执行。

<!--more-->

### asyncio + async/await

为了简化并更好地标识异步IO，从Python3.5开始引入了新的语法async和await，可以让coroutine的代码更简洁易读。请注意，async和await是coroutine的新语法，使用新语法只需要做两步简单的替换：

- 把@asyncio.coroutine替换为async
- 把yield from替换为await



```
#-*- coding:utf8 -*-000
import asyncio

async def test(i):
    print('test_1', i)
    await asyncio.sleep(1)
    print('test_2', i)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    tasks = [test(i) for i in range(3)]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
```

## Gevent

Gevent是一个基于Greenlet实现的网络库，通过greenlet实现协程。基本思想是一个greenlet就认为是一个协程，当一个greenlet遇到IO操作的时候，比如访问网络，就会自动切换到其他的greenlet，等到IO操作完成，再在适当的时候切换回来继续执行。由于IO操作非常耗时，经常使程序处于等待状态，有了gevent为我们自动切换协程，就保证总有greenlet在运行，而不是等待IO操作。

```
#-*- coding:utf8 -*-
import gevent

def test(n):
    for i in range(n):
        print(gevent.getcurrent(), i)

if __name__ == '__main__':
    g1 = gevent.spawn(test, 3)
    g2 = gevent.spawn(test, 3)
    g3 = gevent.spawn(test, 3)

    g1.join()
    g2.join()
    g3.join()
```

可以看到3个greenlet是依次运行而不是交替运行。要让greenlet交替运行，可以通过`gevent.sleep()`交出控制权:

```
def test(n):
    for i in range(n):
        print(gevent.getcurrent(), i)
        gevent.sleep(1)
```

当然在实际的代码里，我们不会用`gevent.sleep()`去切换协程，而是在执行到IO操作时gevent会自动完成，所以gevent需要将Python自带的一些标准库的运行方式由阻塞式调用变为协作式运行。这一过程在启动时通过monkey patch完成：

```
#-*- coding:utf8 -*-
from gevent import monkey; monkey.patch_all()
from urllib import request
import gevent

def test(url):
    print('Get: %s' % url)
    response = request.urlopen(url)
    content = response.read().decode('utf8')
    print('%d bytes received from %s.' % (len(content), url))

if __name__ == '__main__':
    gevent.joinall([
        gevent.spawn(test, 'http://httpbin.org/ip'),
        gevent.spawn(test, 'http://httpbin.org/uuid'),
        gevent.spawn(test, 'http://httpbin.org/user-agent')
    ])
```



```python
def consumer():
    print("[CONSUMER] start")
    r = 'start'
    while True:
        n = yield r
        if not n:
            print("n is empty")
            continue
        print("[CONSUMER] Consumer is consuming %s" % n)
        r = "200 ok"


def producer(c):
    # 启动generator
    start_value = c.send(None)
    print(start_value)
    n = 0
    while n < 3:
        n += 1
        print("[PRODUCER] Producer is producing %d" % n)
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)
    # 关闭generator
    c.close()


# 创建生成器
c = consumer()
# 传入generator
producer(c)
```

```python
# 子生成器
def test(n):
    i = 0
    while i < n:
        yield i
        i += 1

# 委派生成器
def test_yield_from(n):
    print("test_yield_from start")
    yield from test(n)
    print("test_yield_from end")


for i in test_yield_from(3):
    print(i)
```



# 参考资料

[Python异步IO操作](https://zhuanlan.zhihu.com/p/95722895)

[Python黑魔法 --- 异步IO（ asyncio） 协程 ](https://www.jianshu.com/p/b5e347b3a17c)

[python 多进程和协程配合使用](https://cloud.tencent.com/developer/article/1590280)

[(21条消息) 实战异步IO框架：asyncio 下篇_王炳明-CSDN博客](https://blog.csdn.net/weixin_36338224/article/details/109288327?spm=1001.2014.3001.5501)

[(21条消息) 深入异步IO框架：asyncio 中篇_王炳明-CSDN博客](https://blog.csdn.net/weixin_36338224/article/details/109282596?spm=1001.2014.3001.5501)

[(21条消息) 初识异步IO框架：asyncio 上篇_王炳明-CSDN博客_异步io框架](https://blog.csdn.net/weixin_36338224/article/details/109282563?spm=1001.2014.3001.5501)

