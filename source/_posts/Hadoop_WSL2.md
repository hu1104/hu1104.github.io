---
title: Hadoop_WSL2
date: 2021-06-28 12:12:35
tags:
---
# 制作镜像
## 1. 拉取Ubuntu镜像

    首先拉取一个Ubuntu最新的镜像作为基础镜像`docker pull ubuntu:latest`,结束后，运行`docker images`,可以看到
![图片](base_ubuntu.png "打开后显示图片")

## 2. 使用Dockerfile构建包含jdk的ubuntu镜像

    去jdk官网下载jdk包，此处下载的为jdk1.8 ** jdk-8u291-linux-x64.tar.gz**, 将下载好的jdk文件移至wsl2环境下，在此目录下新建Dockerfile文件，并进入编辑状态

```
vim Dockerfile
```

<!--more-->
&emsp;&emsp;在Dockerfile中输入以下内容：

```
FROM ubuntu:latest
MAINTAINER duanmu
ADD jdk-8u291-linux-x64.tar.gz /usr/local/
ENV JAVA_HOME /usr/local/jdk1.8.0_291
ENV CLASSPATH $JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
ENV PATH $PATH:$JAVA_HOME/bin
```

&emsp;&emsp;编辑后保存，开始build镜像

```
docker build -t jdk-20210127 .
```

## 3. 进入镜像

&emsp;&emsp;新建一个以jdk-20210127为基础镜像的容器命名为ubuntu_hadoop并指定容器的hostname为charlie,并进入容器。

```
docker run -it --name=ubuntu_hadoop -h charlie jdk-20210127
```

## 4. 升级apt-get

```
apt-get update
```

## 5. 安装vim

```
apt-get install vim
```

## 6. 更新apt-get镜像源
&emsp;&emsp;默认的apt-get下载源速度太慢，更换下载源可以提升速度,进入下载源列表文件，按a进入insert模式

```
vim /etc/apt/sources.list
```
将其中内容全部替换为

```
deb-src http://archive.ubuntu.com/ubuntu focal main restricted #Added by software-properties
deb http://mirrors.aliyun.com/ubuntu/ focal main restricted
deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted multiverse universe #Added by software-properties
deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted
deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted multiverse universe #Added by software-properties
deb http://mirrors.aliyun.com/ubuntu/ focal universe
deb http://mirrors.aliyun.com/ubuntu/ focal-updates universe
deb http://mirrors.aliyun.com/ubuntu/ focal multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-updates multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse #Added by software-properties
deb http://archive.canonical.com/ubuntu focal partner
deb-src http://archive.canonical.com/ubuntu focal partner
deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted
deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted multiverse universe #Added by software-properties
deb http://mirrors.aliyun.com/ubuntu/ focal-security universe
deb http://mirrors.aliyun.com/ubuntu/ focal-security multiverse
```

## 7. 重新升级apt-get

```
apt-get update
```

## 8. 安装wget

```
apt-get install wget
```

## 9. 创建并进入安装hadoop的文件目录

```
mkdir -p soft/apache/hadoop/
cd soft/apache/hadoop
```

## 10. 通过wget下载安装Hadoop安装包

```
wget http://mirrors.ustc.edu.cn/apache/hadoop/common/hadoop-3.3.0/hadoop-3.3.0.tar.gz
```

## 11. 解压hadoop

```
tar -xvzf Hadoop-3.3.0.tar.gz
```

## 12. 配置环境变量并重启配置文件

```
vim ~/.bashrc
```

新增以下环境变量：

```
export JAVA_HOME=/usr/local/jdk1.8.0_291
export HADOOP_HOME=/soft/apache/hadoop/hadoop-3.3.0
export HADOOP_CONFIG_HOME=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
export PATH=$PATH:$HADOOP_HOME/sbin
```

并重启配置文件

```
source ~/.bashrc
```

## 13. 创建文件夹并修改配置文件

```
cd $HADOOP_HOME
mkdir tmp
mkdir namenode
mkdir datanode
```

修改配置文件

```
cd $HADOOP_CONFIG_HOME
vim core-site.xml
```

将下面内容替换

```
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<!--
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->

<!-- Put site-specific property overrides in this file. -->
<configuration>
<property>
        <name>hadoop.tmp.dir</name>
        <value>/soft/apache/hadoop/hadoop-3.3.0/tmp</value>
</property>
<property>
        <name>fs.default.name</name>
        <value>hdfs://master:9000</value>
        <final>true</final>
</property>
</configuration>
```

更改hdfs-site.xml

```
vim hdfs-site.xml
```

用下面配置替换：

```
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<!--
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->

<!-- Put site-specific property overrides in this file. -->

<configuration>
<property>
        <name>dfs.replication</name>
        <value>2</value>
        <final>true</final>
</property>
<property>
        <name>dfs.namenode.name.dir</name>
        <value>/soft/apache/hadoop/hadoop-3.3.0/namenode</value>
        <final>true</final>
</property>
<property>
        <name>dfs.datanode.name.dir</name>
        <value>/soft/apache/hadoop/hadoop-3.3.0/datanode</value>
        <final>true</final>
</property>
</configuration>
```

接下来

```
vim mapred-site.xml
```

使用下面内容替换

```
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<!--
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->

<!-- Put site-specific property overrides in this file. -->

<configuration>
<property>
        <name>mapred.job.tarcker</name>
        <value>master:9001</value>
</property>
</configuration>
```

## 14. 修改hadoop环境变量

在hadoop的安装目录下，找到`hadoop-env.sh`文件

```
vim hadoop-env.sh
```

在最后添加

```
export JAVA_HOME=/usr/local/jdk1.8.0_291
```

刷新

```
hadoop namenode -format
```

## 15. 安装SSH

hadoop的环境必须满足ssh免密登陆，先安装ssh

```
apt-get install net-tools
apt-get install ssh
```

创建sshd目录

```
mkdir -p ~/var/run/sshd
```

生成访问密钥

```
cd ~/
ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
cd .ssh
cat id_rsa.pub >> authorized_keys
```

这一步骤提示安装路径与设置密码时全布直接按回车即可设置成免密。

### 修改ssh配置

```
vim /etc/ssh/ssh_config
```

添加,将下面这句话直接添加即可，也可以在文件中找到被注释的这句话去修改。

```
StrictHostKeyChecking no #将ask改为no
```

```
vim etc/ssh/sshd_config
```

在末尾添加：

```
#禁用密码验证
PasswordAuthentication no
#启用密钥验证
RSAAuthentication yes
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
```

最后使用下面语句测试是否免密登陆，
```
ssh localhost
```

当出现报错时，考虑输入：

```
 /etc/init.d/ssh restart
```

为了避免踩坑，先提前设置,进入环境变量

```
vim /etc/profile
```

增加如下内容并保存：

```
export HDFS_NAMENODE_USER=root
export HDFS_DATANODE_USER=root
export HDFS_SECONDARYNAMENODE_USER=root
export YARN_RESOURCEMANAGER_USER=root
export YARN_NODEMANAGER_USER=root
```

使配置生效

```
source /etc/profile
```

## 16. 导出镜像

至此镜像已经配置完成，退出容器，将配置好的镜像保存，其中xxxx为刚刚操作的容器的id，可以使用docker ps -a查看

```
docker commit xxxx ubuntu:hadoop
```

此时ubuntu_hadoop就是最终配置好的包含hadoop的镜像。

## 17. 集群测试

依次构建并启动三个以刚刚生成的镜像为基本镜像的容器，依次命名为master 、slave1、slave2，并将master做端口映射（提示：容器要处于运行状态，生成容器后使用ctrl+P+Q退出可以使容器保持后台运行。）

```
docker run -it  -h master --name=master -p 9870:9870 -p 8088:8088 -p 9000:9000 ubuntu:hadoop 
docker run -it  -h slave1 --name=slave1 ubuntu:hadoop 
docker run -it  -h slave2 --name=slave2 ubuntu:hadoop 
```

修改每个容器的host文件
对matser、slave1、slave2里的host文件，分别加入其他两个容器的ip

```
vim /etc/hosts
```

```
127.0.0.1       localhost
::1     localhost ip6-localhost ip6-loopback
fe00::0 ip6-localnet
ff00::0 ip6-mcastprefix
ff02::1 ip6-allnodes
ff02::2 ip6-allrouters
172.17.0.2      master
172.17.0.3      slave1
172.17.0.4      slave2 ###根据实际修改
```

### 修改master中slaves文件

注意，在hadoop3.3.0版本中并不是修改slaves文件，而是修改workers文件。此处为3.3.0版本的一些变化。
老版本（自行查找hadoop版本中已存在文件是slaves还是iworkers）

```
cd $HADOOP_CONFIG_HOME/
vim workers
```

将其他两个节点名称加入文件

```
slave1
slave2
```

### 启动hadoop

此时报错的话，可在每个节点运行

```
source /etc/profile
```



# Reference

1. https://blog.csdn.net/weixin_43993764/article/details/113405025
2. https://blog.csdn.net/u013015629/article/details/70045809

<!--
&emsp;&emsp;<font face="黑体" size=10>16. 我是黑体字</font>
-->

<div style="display:none">这是一段注释</div>