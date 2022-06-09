---
title: decorator
date: 2021-08-20 09:05:48
tags:
categories:
password:
abstract:
message: 
---

<!-- toc -->





# 什么是装饰器

一个decorator只是一个带有一个函数作为参数并返回一个替换函数的闭包。
简单示例：

```
# 未使用装饰器时
def outer(some_func):
	def inner():
		print ("before some_func")
		ret = some_func() # 1
		return ret + 1
	return inner
def foo():
	return 1
decorated = outer(foo)
decorated()

#使用装饰器时
def outer(some_func):
	def inner():
		print ("before some_func")
		ret = some_func() # 1
		return ret + 1
	return inner
@outer
def foo():
	return 1

foo()
```

<!--more-->

# 简单装饰器



引入装饰器会便于开发，便于代码复用

简单示例：

```
def say_hello():
    print("[DEBUG]: enter say_hello()")
    print("hello!")

def say_goodbye():
    print( "[DEBUG]: enter say_goodbye()")
    print( "hello!")
    
if __name__ == '__main__':
    say_hello()
    say_goodbye()
```

进一步

```
def debug():
    import inspect
    caller_name = inspect.stack()[1][3]
    print("[DEBUG]: enter {}()".format(caller_name)  ) 

def say_hello():
    debug()
    print("hello!")

def say_goodbye():
    debug()
    print( "goodbye!")

if __name__ == '__main__':
    say_hello()
    say_goodbye()
```

但依然每个业务函数都需要调用一次`debug()`函数

使用装饰器:

```
def debug(func):
    def wrapper():
        print("[DEBUG]: enter {}()".format(func.__name__))
        return func()
    return wrapper

@debug
def say_hello():
    print( "hello!")

```

这个装饰器依然存在一个问题， 不能引入参数

```
def debug(func):
    def wrapper(something):  # 指定一毛一样的参数
        print "[DEBUG]: enter {}()".format(func.__name__)
        return func(something)
    return wrapper  # 返回包装过函数

@debug
def say(something):
    print "hello {}!".format(something)
```

那么如果参数不定的时候呢？`(*args, **kwargs)`就可以派上用场了，其中`*args`只是表明在函数定义中位置参数应该保存在变量`args`中, *表示`iterables`和位置参数,**表示dictionaries & key/value对

```
def debug(func):
    def wrapper(*args, **kwargs):  # 指定宇宙无敌参数
        print "[DEBUG]: enter {}()".format(func.__name__)
        print 'Prepare and say...',
        return func(*args, **kwargs)
    return wrapper  # 返回

@debug
def say(something):
    print "hello {}!".format(something)
```

至此，简单的装饰器完成！



# 高级装饰器

## 基于函数的带参装饰器

示例：

```
def logging(level): # 传递装饰器参数
    def wrapper(func): # 调用函数名
        def inner_wrapper(*args, **kwargs): # 函数参数
            print "[{level}]: enter function {func}()".format(
                level=level,
                func=func.__name__)
            return func(*args, **kwargs)
        return inner_wrapper
    return wrapper

@logging(level='INFO')
def say(something):
    print "say {}!".format(something)

# 如果没有使用@语法，等同于
# say = logging(level='INFO')(say)

@logging(level='DEBUG')
def do(something):
    print "do {}...".format(something)

if __name__ == '__main__':
    say('hello')
    do("my work")
```

你可以这么理解，当带参数的装饰器被打在某个函数上时，比如`@logging(level='DEBUG')`，它其实是一个函数，会马上被执行，只要这个它返回的结果是一个装饰器时，那就没问题。



## 基于类的不带参装饰器

装饰器函数其实是这样一个接口约束，它必须接受一个callable对象作为参数，然后返回一个callable对象。在Python中一般callable对象都是函数，但也有例外。只要某个对象重载了`__call__()`方法，那么这个对象就是callable的。



装饰器要求接受一个callable对象，并返回一个callable对象（不太严谨，详见后文）。那么用类来实现也是也可以的。我们可以让类的构造函数`__init__()`接受一个函数，然后重载`__call__()`并返回一个函数，也可以达到装饰器函数的效果。

 ```
 class logging(object):
     def __init__(self, func): # 接受函数
         self.func = func
 
     def __call__(self, *args, **kwargs): # 函数参数
         print "[DEBUG]: enter function {func}()".format(
             func=self.func.__name__)
         return self.func(*args, **kwargs)
 @logging
 def say(something):
     print "say {}!".format(something)
 ```



## 基于类的带参数装饰器

```
class logging(object):
    def __init__(self, level='INFO'):
        self.level = level
        
    def __call__(self, func): # 接受函数
        def wrapper(*args, **kwargs):
            print "[{level}]: enter function {func}()".format(
                level=self.level,
                func=func.__name__)
            func(*args, **kwargs)
        return wrapper  #返回函数

@logging(level='INFO')
def say(something):
    print "say {}!".format(something)
```



# 内置装饰器



## @classmethod

类方法，不需要实例化，也不需要self参数，需要一个cls参数，可以用类名调用，也可以用对象来调用。



原则上，类方法是将类本身作为对象进行操作的方法。假设有个方法，且这个方法在逻辑上采用类本身作为对象来调用更合理，那么这个方法就可以定义为类方法

```
class A:
    """docstring for A"""
    # 类变量v
    v = 0
 
    def __init__(self):
        # __init__定义的为实例变量，属于类的实例
        self.my_v = 10000000
 
    # 类方法需要使用@classmethod装饰器定义
    @classmethod
    # 类方法至少有一个形参,第一个形参用于绑定类,约定为:'cls'
    def get_v(cls):
        """此方法为类方法,cls用于绑定调用此方法的类;此方法用于返回类变量v的值"""
        return cls.v
 
    @classmethod
    def set_v(cls, value):
        cls.v = value
 
 
if __name__ == "__main__":
    # 通过类实例来调用类方法
    print(A.get_v())
    A.set_v(100)
    print(A.get_v())
 
    # 通过对象实例调用类方法
    a = A()
    print(a.get_v())
    a.set_v(200)
    print(a.get_v())
 
    # 访问实例属性
    print(a.my_v)
```



## @staticmethod

静态方法，不需要实例化，不需要self和cls等参数，就跟使用普通的函数一样，只是封装在类中

静态方法是类中的函数，不需要实例。静态方法主要是用来存放逻辑性的代码，逻辑上属于类，但是和类本身没有关系，也就是说在静态方法中，不会涉及到类中的属性和方法的操作。可以理解为，静态方法是个独立的、单纯的函数，它仅仅托管于某个类的名称空间中，便于使用和维护。

```
class Student:
    """描述学生的信息"""
    count = 0
 
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score
        self.__class__.count += 1
 
    def print_info(self):
        print("{}: age={}, score={}".format(self.name, self.age, self.score))
 
    @classmethod
    def get_stu_number(cls):
        """只访问类变量，使用类方法即可"""
        return cls.count
 
    @staticmethod
    def average(students, kind):
        """不需要访问实例变量和类变量，仅仅是定义在类内的函数，使用静态方法即可"""
        sum_kind = 0
        for student in students:
            sum_kind += student.__dict__[kind]
        average = sum_kind // Student.get_stu_number()
        return average
 
    @staticmethod
    def add_stu_info():
        """一次性录入所有的学生信息，并以列表形式返回所有创建好的学生实例"""
        students = []
        while True:
            name = input('输入姓名:') or 'q'
            if name == 'q':
                break
            age = int(input('输入年龄:'))
            score = int(input('输入成绩:'))
            student = Student(name, age, score)
            students.append(student)
        return students
 
    @classmethod
    def remove_student(cls, name, students):
        """根据姓名删除列表中的学生"""
        for student in students:
            if student.name.lower() == name.lower():
                stu_list.remove(student)
                cls.count -= 1

>> Student.average(stu_list, 'score')
95
>> Student.average(stu_list, 'age')
13
```



## @property

属性方法，主要作用是将一个操作方法封装成一个属性,用户用起来就和操作普通属性完全一致,非常简单.定义时，在实例方法的基础上添加@property装饰器，并且只有一个self参数，调用时，不需要括号

@property 是经典类中的一种装饰器，新式类中具有三种:

1. @property获取属性
2. @方法名.setter 修改属性
3. @方法名.deleter 删除属性
   

```
class Goods(object):

    def __init__(self):
        # 原价
        self.original_price = 100
        # 折扣
        self.discount = 0.8

    @property
    def price(self):
        # 实际价格 = 原价*折扣
        new_price = self.original_price*self.discount
        return new_price

    @price.setter
    def price(self,value):
        self.original_price = value

    @price.deleter
    def price(self):
        del self.original_price


obj = Goods()
# print(obj.price)
obj.price = 200
print(obj.price)
del obj.price  # 删除了类中的price属性若再次调用就会报错

```



## @wraps

Python装饰器（decorator）在实现的时候，被装饰后的函数其实已经是另外一个函数了（函数名等函数属性会发生改变），为了不影响，Python的functools包中提供了一个叫wraps的decorator来消除这样的副作用。写一个decorator的时候，最好在实现之前加上functools的wrap，它能保留原有函数的名称和docstring。

不加wraps:

```
#coding=utf-8
# -*- coding=utf-8 -*- 
from functools import wraps   
def my_decorator(func):
    def wrapper(*args, **kwargs):
        '''decorator'''
        print('Calling decorated function...')
        return func(*args, **kwargs)
    return wrapper  
 
@my_decorator 
def example():
    """Docstring""" 
    print('Called example function')
print(example.__name__, example.__doc__)

# 输出
('wrapper', 'decorator')
[Finished in 0.2s]
```

加上后：

```
#coding=utf-8
# -*- coding=utf-8 -*- 
from functools import wraps   
def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        '''decorator'''
        print('Calling decorated function...')
        return func(*args, **kwargs)
    return wrapper  
 
@my_decorator 
def example():
    """Docstring""" 
    print('Called example function')
print(example.__name__, example.__doc__)
# 输出
('example', 'Docstring')
[Finished in 0.5s]
```



## python内置类属性

1. `__dict__ `: 类的属性（包含一个字典，由类的数据属性组成）

2. `__doc__` :类的文档字符串

3. `__name__`: 类名

4. `__module__`: 类定义所在的模块（类的全名是'__main__.className'，如果类位于一个导入模块mymod中，那么className.__module__ 等于 mymod）

5. `__bases__` : 类的所有父类构成元素（包含了一个由所有父类组成的元组）



# 参考资料

[详解Python的装饰器](https://www.cnblogs.com/cicaday/p/python-decorator.html)

[函数式编程入门](http://ruanyifeng.com/blog/2017/02/fp-tutorial.html)

[装饰器八种写法](https://zhuanlan.zhihu.com/p/269012332)

