---
title: yolox训练自定义数据集
date: 2021-08-30 14:13:04
tags:
categories:
password:
abstract:
message:
---



# 环境配置

 第一步：安装YOLOX

```
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
```

<!--more-->

第二步：安装apex

```
# 如果不想训练模型，可跳过这步。
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

```

此处需要torch的cuda版本和外部环境的cuda版本一致，比如我们的服务器是10.2，那么torch的也要是10.2

可通过`torch.version.cuda`确认，最好是选择docker来配置。Windows也不推荐，即使是wsl。



第三步： 安装 [pycocotools](https://github.com/cocodataset/cocoapi)

```
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```



# 修改配置

修改`exps/example/yolox_voc/yolox_voc_s.py`, 主要涉及类别和文件路径

![在这里插入图片描述](https://img-blog.csdnimg.cn/img_convert/fb863c7fbfa25392b426aa319026887b.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/img_convert/b52f331bb2f1f35a057136c8e5b77656.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/img_convert/febec4ac7bf5e614f383ea5932007ffe.png)



修改`yolox/data/datasets/__init__.py`

![img](https://img-blog.csdnimg.cn/img_convert/e4575a30f261ae0371fd460684d41763.png)

修改`yolox/data/datasets/voc_classes.py`

![image-20210830143808609](image-20210830143808609.png)



# 开始训练

`python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 16 --fp16 -o -c yolox_s.pth`

- -d 使用多少张显卡训练
- -b 批次大小
- --fp16 是否开启半精度训练



# 批量推理

非官方实现，其实也就是for循环实现的。

`tools/test_imgs.py`:

![image-20210830144401995](image-20210830144401995.png)

![image-20210830144449907](image-20210830144449907.png)



# 参考资料

[深入浅出Yolox之自有数据集训练超详细教程 ](https://zhuanlan.zhihu.com/p/397499216)

[YOLOX自定义数据集训练](https://blog.csdn.net/qq_39056987/article/details/119002910)

[DataXujing/YOLOX-: YOLOX 训练自己的数据集 TensorRT加速 详细教程](https://github.com/DataXujing/YOLOX-)

