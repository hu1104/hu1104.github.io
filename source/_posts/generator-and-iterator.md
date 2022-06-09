---
title: generator_and_iterator
date: 2021-08-20 11:14:02
tags:
categories:
password:
abstract:
message:
---

<!--toc-->

# 列表表达式

```
info = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
a = [i+1 for i in range(10)]
print(a)
```



多重循环

```
a = [(i,j) for i in range(4) for j in range(2)]
print(a)
```



多重循环+判断语句

```
ls = [('x',['open1','open1','open1']),('y',['open1','open1','open0']),('z',['open0','open0','open1'])]

def find(con):
    return [j for i in con for j in i if j.isdigit()]
 
[w for w,con in ls if find(con)==['1','1','1']]
```

<!--more-->

# 生成器

1. 什么是生成器

   生成器是一种特殊的迭代器，生成器的出现是为了简化迭代器应用。

   生成器的主要思想：对于可以公式自动生成的数字序列，由计算机不断迭代，每次只生成一个数字，从而通过循环遍历生成序列中的所有元素。所以说，生成器产生的不是一个静态的值（比如类似字符串、元组和列表等，都是一次性生成所有值），而是一个动态的数据流。

   示例1：

   ```
   a = (x**2 for x in range(1,9))
   print(type(a))
   next(a)
   ```

   示例2：

   ```
   def gen(x):
   	x += 1
   	yield x**2
   b = gen(0)
   print(type(b))
   next(b)
   ```

   可以看到，变量a和b都是生成器，我们不能直接使用a、b，因为它们实际上保存的是一个公式，使用时可以调用内置函数next()，由next(a)、next(b)来动态生成序列中的下一个值。采用生成器的好处是：节省内存空间，特别是对于数据量大的序列，一次性生成所有值将会耗费大量内存，而采用生成器可以极大地节省存储空间。同时，生成器还可以处理无限长的序列。比如，上述实例中，变量b就是一个无限序列，理论上可以永远next(b)，而且每次都是按顺序生成其中的一个值。

   

   可以把生成器看作是一种特殊的函数，它与一般函数最主要的区别就在于生成器函数中有关键字yield。比如，上述实例2，函数中只要有yield关键字，就是一个生成器函数。

2. 生成器怎么用

   ​	(1). 生成器使用场景

   ​			当你需要生成一个大型的序列，但又不想因此占用大量的存储空间，提高存储和计算效率。此时，可以考虑用生成器。

   ​	(2). 生成器的构造

   ​			主要有两种方式：一是生成器表达式；二是生成器函数。上面实例1就是生成器表达式；实例2其实就是生成器函数。

   ​	(3). 生成器的使用

   ​			①采用for循环

   ​			②采用内置函数next()遍历生成器元素

   ​			③采用生成器自身方法`__next()__`循环生成下一个值。

# 迭代器

1. 什么是迭代器

   ​	首先了解几个概念：

   ​		(1). 可迭代对象。可以直接作用于for 循环的对象统称为可迭代对象：Iterable。可以使用isinstance()判断一个对象是否为可Iterable对象。

   ```
   >>> from collections import Iterable
   >>> isinstance(fib(8), Iterable)
   True
   >>> b = [1,2,3,4]
   >>> isinstance(b, Iterable)
   True
   >>> c = 8
   >>> isinstance(c, Iterable)
   False
   ```

   Python中可直接采用for循环的对象有：一类是集合数据类型，如list，tuple，dict，set，str等；一类是generator，包括生成器表达式和带yield的生成器函数。

   ​		(2). 迭代器。Python中一个实现了_iter_方法和_next_方法的类对象，就是迭代器。

    	   (3).迭代器协议：要构造一个迭代器，对象需要提供next()方法，它要么返回迭代中的下一项，要么就引起一个StopIteration异常，以终止迭代。

2. 迭代器的构造

   （1）自定义迭代器类

```
class Fib(object):
    def __init__(self, max):
        super(Fib, self).__init__()
        self.max = max
    def __iter__(self):
        self.a = 0
        self.b = 1
        return self
    def __next__(self):
        fib = self.a
        if fib > self.max:
            raise StopIteration
        self.a, self.b = self.b, self.a + self.b
        return fib
# 定义一个main函数，循环遍历每一个菲波那切数
def main():
    # 20以内的数
    fib = Fib(20)
    for i in fib:
        print(i)
# 测试
if __name__ == '__main__':
    main()
```

​	（2）通过调用内置函数iter()构造迭代器

```
>>> g = iter(range(10))
>>> isinstance(g, Iterator)
True
>>> isinstance(range(10), Iterable)
True
>>> isinstance(range(10), Iterator)
False
```

注意，不少文章中写道，map、filter等内置函数返回的都是生成器，还有个别资料中说range返回的也是生成器。要检查一个对象是否为迭代器，也可以采用isinstance()判断，所以我们可以进行以下的判断：

```
>>> from collections import Iterable， Iterator， Generator
>>> d= range(10)
>>> isinstance(d, Iterable)
True
>>> isinstance(d, Iterator)
False
>>> e = map(lambda x : x**2, [1,2,4,6,7,8,9])
>>> isinstance(e, Iterable)
True
>>> isinstance(e, Iterator)
True
>>> isinstance(e, Generator)
>>> f = (x**2 for x in [1,2,4,6,7,8,9])
>>> isinstance(f, Iterator)
True
>>> isinstance(f, Generator)
True
```



# 参考资料

[生成器和迭代器总结](https://zhuanlan.zhihu.com/p/122537818)

