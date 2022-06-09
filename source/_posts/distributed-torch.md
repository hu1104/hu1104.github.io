---
title: distributed_torch
date: 2021-08-26 16:11:12
tags:
categories:
password:
abstract:
message:
---



# 多GPU分配

实现逻辑：寻找（可用显存 / 总显存）最大的的GPU，并优先安排任务

nvidia-smi可以很方便的获得GPU的各种详细信息。

首先获得可用的GPU数目，nvidia-smi -L | grep GPU |wc -l

然后获得GPU各自的总显存，nvidia-smi -q -d Memory | grep -A4 GPU | grep Total | grep -o '[0-9]\+'

最后获得GPU各自的可用显存，nvidia-smi -q -d Memory | grep -A4 GPU | grep Free | grep -o '[0-9]\+'

将（可用显存 / 总显存）另存为numpy数组，并使用np.argmax返回值即为可用GPU

```
def available_GPU(self):
    import subprocess
    import numpy as np
    nDevice = int(subprocess.getoutput("nvidia-smi -L | grep GPU |wc -l"))
    total_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total | grep -o '[0-9]\+'")
    total_GPU = total_GPU_str.split('\n')
    total_GPU = np.array([int(device_i) for device_i in total_GPU])
    avail_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free | grep -o '[0-9]\+'")
    avail_GPU = avail_GPU_str.split('\n')
    avail_GPU = np.array([int(device_i) for device_i in avail_GPU])
    avail_GPU = avail_GPU / total_GPU
    return np.argmax(avail_GPU)
```



# 参考资料

[pytorch(分布式)数据并行个人实践总结——DataParallel/DistributedDataParallel ](https://www.cnblogs.com/yh-blog/p/12877922.html)

[torch 多进程训练(详细例程)](https://blog.csdn.net/junqing_wu/article/details/112732338)

[PyTorch多进程分布式训练最简单最好用的实施办法](https://blog.csdn.net/qq_34914551/article/details/110576421)

[python并行编程 - GPU篇](https://blog.csdn.net/ZAQ1018472917/article/details/84626040)

[Pytorch 分布式、多进程模块测试 ](https://zhuanlan.zhihu.com/p/107230545)

[多进程GPU调用问题](https://blog.csdn.net/baidu_36669549/article/details/95094464)

[GPU加速02:超详细Python Cuda零基础入门教程，没有显卡也能学](https://zhuanlan.zhihu.com/p/77307505)

[使用python多GPU任务分配](https://blog.csdn.net/sh39o/article/details/90382101)

[Deep Learning:PyTorch 基于docker 容器的分布式训练实践](https://blog.csdn.net/github_37320188/article/details/100519346)

