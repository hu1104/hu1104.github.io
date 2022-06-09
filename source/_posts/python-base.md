---
title: python_base
date: 2021-08-30 10:18:47
tags:
categories:
password:
abstract:
message:
---





# __init__.py

首先，module其实就是一个.py文件，中文名为模块，其内置有各种函数和类与变量等。而package就是一个包含很多subpackage或者module(.py文件)的一个包。

![img](https://upload-images.jianshu.io/upload_images/16486710-40ddf50fd0d6b105.jpg?imageMogr2/auto-orient/strip|imageView2/2/format/webp)

一个directories 只有包含__init__.py文件才会被python识别成package。只有在import package时，才会执行package目录下的__init__.py文件。

若文件结构如下：

```
mypackage
    ——__init__.py
    ——subpackage_1
        ——__init__.py
        ——test11.py
        ——test12.py
    ——subpackage_2
        ——__init__.py
        ——test21.py
        ——test22.py
```



# pickle, json

都是四个函数：

```
pickle.dumps()：将 Python 中的对象序列化成二进制对象，并返回；
pickle.loads()：读取给定的二进制对象数据，并将其转换为 Python 对象；
pickle.dump()：将 Python 中的对象序列化成二进制对象，并写入文件；
pickle.load()：读取指定的序列化数据文件，并返回对象。


json.load()从文件中读取json字符串
json.loads()将json字符串转换为字典类型
json.dumps()将python中的字典类型转换为字符串类型
json.dump()将json格式字符串写到文件中
```



# 参考资料



[__init__.py文件与__all__变量](https://www.jianshu.com/p/eaae9678a779)

[python模块中__init__.py的作用](https://blog.csdn.net/yucicheung/article/details/79445350)

[Python：__init__.py文件和、__all__、import、__name__、__doc__ ](https://www.cnblogs.com/qi-yuan-008/p/12827918.html)

