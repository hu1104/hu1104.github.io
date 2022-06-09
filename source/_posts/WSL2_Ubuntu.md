---
title: WSL2安装Ubuntu
date: 2021-06-25 09:22:46
tags: 
  - win10
  - ubuntu
categories:
  - 教程
---

# 系统更新至预览版

1. 加入Windows Insider Program

    [链接](https://insider.windows.com/en-us/getting-started)

    ![图片](wip.png "打开后显示")

2. 注册加入完成后，前往【开始】菜单>【设置】>【更新和安全】>【Windows 预览体验计划】，选择【开始】，然后点击【确认】。

    ![图片](update.png "打开后显示")
    ![图片](choose.png "打开后显示")
    ![图片](microsoft_login.png "打开后显示")

3. 漫长的等待，等待过程中如果出现太久旋转没动静，可反复取消-选择账户，几次下来一般就可以了。随后使用加入Windows Insider Program时注册的账号密码，之后出现
   
    ![图片](dev.png "打开后显示")
    选择dev渠道，然后立即重启。

4. 前往【开始】菜单>【设置】>【更新和安全】>【Windows 更新】，下载更新完window，
选择【检查更新】，然后耐心等待最新 Windows 10 预览版的下载和安装。

    ![图片](update_wid.png "打开后显示")

5. 检查系统更新成功与否
   
    win+R->winver:
    ![图片](version.png "打开后显示")
    确认这里的os内部版本和第四步下载的版本一致
    
    <!--more-->

# 安装驱动

1. 前往[链接](https://developer.nvidia.com/cuda/wsl)
   
    ![图片](driver.png "打开后显示")

2. 下载完成后，正常win程序安装

# 安装WSL2

按照官方说法，使用预览版会有简单操作，但我是在未使用预览版时安装的，所以仅供参考。


1. 管理员身份运行power shell

    如果出现以下问题：
    ![图片](problem.png "打开后显示")
    则执行`set-ExecutionPolicy RemoteSigned`
    成功结果为：
    ![图片](success.png "打开后显示")
    如果没有出现上述问题，则执行`dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`开启子系统
    ![图片](subsystem.png "打开后显示")
    然后，执行`dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart`开启虚拟机
    ![图片](vm.png "打开后显示")
    <font color=red>随后重启计算机！</font>

2. 安装WSL内核更新包[链接](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)

3. 将wsl2设置为默认版本：`set --set-default-version 2`

# 安装ubuntu18.04

1. 打开Microsoft store,搜索ubuntu18,点击获取，等待下载安装，安装完成后，点击右上角的启动按钮，会出现下面操作；设置好用户名及密码，子系统安装完成。
![图片](ubuntu.png "打开后显示")

2. 安装完成后，运行`wsl --list --verbose`
![图片](wsl_lv.png "打开后显示")

3. 运行
   
    + 点击ubuntu图标
        ![图片](run_1.png "打开后显示")
    + 在power shell中运行`wsl`
        ![图片](run_2.png "打开后显示")

# 安装cuda toolkit(在ubuntu环境下)

1. 配置cuda网络仓库
    依次执行
    ```
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    
    sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
    
    sudo apt-get update
    ```

2. 安装cuda
   
    执行`sudo apt-get install -y cuda-toolkit-11-0`

# 安装docker和nvidia-docker

1. 安装docker-ce:`curl https://get.docker.com | sh`

2. 安装nvidia-docker2:
    ```
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container-experimental.list | sudo tee /etc/apt/sources.list.d/libnvidia-container-experimental.list
    
    sudo apt-get update
    
    sudo apt-get install -y nvidia-docker2
    
    sudo gpasswd -a username docker ###username 为设置的Ubuntu用户名
    
    newgrp docker
    ```

3. 完成安装
    ```
    sudo service docker stop
    
    sudo service docker start
    ```

4. 验证安装
    执行`docker run --runtime=nvidia  --rm -it --name tensorflow-1.14.0 tensorflow/tensorflow:1.14.0-gpu-py3`

    然后执行
    ```
    python
    import tensorflow as tf
    print(tf.test.is_gpu_available()) ###输出True
    ```

#  Reference

1. [CUDA on WSL :: CUDA Toolkit Documentation (nvidia.com)](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)