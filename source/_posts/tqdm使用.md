---
title: tqdm使用
date: 2021-08-25 16:22:26
tags:
categories:
password:
abstract:
message:
---

# 基本使用

1. 迭代的形式

   使用`tqdm()`封装可迭代的对象：

   ```
   from tqdm import tqdm
   from time import sleep
   
   text = ""
   for char in tqdm(["a", "b", "c", "d"]):
       sleep(0.25)
       text = text + char
   ```

   <!--more-->

2. 手动的形式

   除了迭代的形式，你可以手动控制进度，加一个tqdm上下文即可：

   ```
   with tqdm(total=100) as pbar:
       for i in range(10):
           sleep(0.1)
           pbar.update(10)
   ```

   上述例子中，pbar 是 tpdm 的“进度”，每一次对 pbar 进行 update 10 都相当于进度加10。

   也可以不是上下文形式：

   ```
   pbar = tqdm(total=100)
   for i in range(10):
       sleep(0.1)
       pbar.update(10)
   pbar.close()
   ```

3. 观察处理的数据

   通过`tqdm`提供的`set_description`方法可以实时查看每次处理的数据

   ```
   from tqdm import tqdm
   import time
   
   pbar = tqdm(["a","b","c","d"])
   for c in pbar:
     time.sleep(1)
     pbar.set_description("Processing %s"%c)
   ```

   

4. linux命令行进度条

   不使用tqdm

   ```
   $ time find . -name '*.py' -type f -exec cat \{} \; | wc -l
   857365
   
   real  0m3.458s
   user  0m0.274s
   sys   0m3.325s
   ```

   使用tqdm

   ```
   $ time find . -name '*.py' -type f -exec cat \{} \; | tqdm | wc -l
   857366it [00:03, 246471.31it/s]
   857365
   
   real  0m3.585s
   user  0m0.862s
   sys   0m3.358s
   ```

   指定tqdm的参数控制进度条

   ```
   $ find . -name '*.py' -type f -exec cat \{} \; |
     tqdm --unit loc --unit_scale --total 857366 >> /dev/null
   100%|███████████████████████████████████| 857K/857K [00:04<00:00, 246Kloc/s]
   
   
   $ 7z a -bd -r backup.7z docs/ | grep Compressing |
     tqdm --total $(find docs/ -type f | wc -l) --unit files >> backup.log
   100%|███████████████████████████████▉| 8014/8014 [01:37<00:00, 82.29files/s]
   ```

5. 自定义进度条显示信息

   通过`set_description`和`set_postfix`方法设置进度条显示信息

   ```
   from tqdm import trange
   from random import random,randint
   import time
   
   with trange(100) as t:
     for i in t:
       #设置进度条左边显示的信息
       t.set_description("GEN %i"%i)
       #设置进度条右边显示的信息
       t.set_postfix(loss=random(),gen=randint(1,999),str="h",lst=[1,2])
       time.sleep(0.1)
   
   ```

   

   ```
   from tqdm import tqdm
   import time
   
   with tqdm(total=10,bar_format="{postfix[0]}{postfix[1][value]:>9.3g}",
        postfix=["Batch",dict(value=0)]) as t:
     for i in range(10):
       time.sleep(0.05)
       t.postfix[1]["value"] = i / 2
       t.update()
   ```

   

6. 多层循环进度条

   通过`tqdm`也可以很简单的实现嵌套循环进度条的展示

   ```
   from tqdm import tqdm
   import time
   
   for i in tqdm(range(20), ascii=True,desc="1st loop"):
     for j in tqdm(range(10), ascii=True,desc="2nd loop"):
       time.sleep(0.01)
   ```

   

7. 多进程进度条

   在使用多进程处理任务的时候，通过tqdm可以实时查看每一个进程任务的处理情况

   ```
   from time import sleep
   from tqdm import trange, tqdm
   from multiprocessing import Pool, freeze_support, RLock
   
   L = list(range(9))
   
   def progresser(n):
     interval = 0.001 / (n + 2)
     total = 5000
     text = "#{}, est. {:<04.2}s".format(n, interval * total)
     for i in trange(total, desc=text, position=n,ascii=True):
       sleep(interval)
   
   if __name__ == '__main__':
     freeze_support() # for Windows support
     p = Pool(len(L),
          # again, for Windows support
          initializer=tqdm.set_lock, initargs=(RLock(),))
     p.map(progresser, L)
     print("\n" * (len(L) - 2))
   ```

   

8. pandas中使用tqdm

   ```
   import pandas as pd
   import numpy as np
   from tqdm import tqdm
   
   df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))
   
   
   tqdm.pandas(desc="my bar!")
   df.progress_apply(lambda x: x**2)
   ```

   

9. 递归使用进度条

   ```
   from tqdm import tqdm
   import os.path
   
   def find_files_recursively(path, show_progress=True):
     files = []
     # total=1 assumes `path` is a file
     t = tqdm(total=1, unit="file", disable=not show_progress)
     if not os.path.exists(path):
       raise IOError("Cannot find:" + path)
   
     def append_found_file(f):
       files.append(f)
       t.update()
   
     def list_found_dir(path):
       """returns os.listdir(path) assuming os.path.isdir(path)"""
       try:
         listing = os.listdir(path)
       except:
         return []
       # subtract 1 since a "file" we found was actually this directory
       t.total += len(listing) - 1
       # fancy way to give info without forcing a refresh
       t.set_postfix(dir=path[-10:], refresh=False)
       t.update(0) # may trigger a refresh
       return listing
   
     def recursively_search(path):
       if os.path.isdir(path):
         for f in list_found_dir(path):
           recursively_search(os.path.join(path, f))
       else:
         append_found_file(path)
   
     recursively_search(path)
     t.set_postfix(dir=path)
     t.close()
     return files
   
   find_files_recursively("E:/")
   ```

10. 注意

    在使用`tqdm`显示进度条的时候，如果代码中存在`print`可能会导致输出多行进度条，此时可以将`print`语句改为`tqdm.write`，代码如下

    ```
    for i in tqdm(range(10),ascii=True):
      tqdm.write("come on")
      time.sleep(0.1)
    ```

    

11. alive-process 花式进度条

[酷炫的 Python 进度条开源库：alive-progress-技术圈 (proginn.com)](https://jishuin.proginn.com/p/763bfbd55bf8)

