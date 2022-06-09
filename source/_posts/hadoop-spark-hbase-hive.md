---
title: hadoop_spark_hbase_hive
date: 2021-07-14 11:25:56
tags:
---



hdfs-site.xml

<!--more-->

```
<?xml version="1.0" encoding="UTF-8"?>
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
                                        <value>/usr/local/hadoop/namenode</value>
                                                        <final>true</final>
                                                </property>
                                                <property>
                                                                <name>dfs.datanode.name.dir</name>
                                                                                <value>/usr/local/hadoop/datanode</value>
                                                                                                <final>true</final>
                                                                                        </property>
                                                                                        <property>
                                                                                                        <name>dfs.namenode.secondary.http-address</name>

<value>master:50090</value>

                </property>
</configuration>
```



core-site.xml

```
<?xml version="1.0" encoding="UTF-8"?>
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
                                        <value>file:/usr/local/hadoop/tmp</value>
                                                        <description>Abase for other temporary directories.</description>
                                                </property>
<property>
           <name>fs.defaultFS</name>
                      <value>hdfs://master:9000</value>
                               </property>

                               <property>
                                                       <name>ha.zookeeper.quorum</name>
                                                                               <value>master:2181,slave1:2181,slave2:2181</value>
                                                                                               </property>
</configuration>
~                                    
```



hadoop-env.sh

```
export JAVA_HOME=/usr/local/jdk1.8.0_291
export HDFS_NAMENODE_USER=root
export HDFS_DATANODE_USER=root
export HDFS_SECONDARYNAMENODE_USER=root
export YARN_RESOURCEMANAGER_USER=root
export YARN_NODEMANAGER_USER=root
```



mapred-site.xml

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
                        <name>mapreduce.framework.name</name>
                                        <value>yarn</value>
                                                    </property>
        <property>
                        <name>mapreduce.jobhistory.address</name>
                                        <value>master:10020</value>
                                </property>
                                <property>
                                                <name>mapreduce.jobhistory.webapp.address</name>
                                                                <value>master:19888</value>
                                                        </property>
</configuration>
~                                                                                                                       ~                                                                                                                       ~   
```



yarn-site.xml

```
<?xml version="1.0"?>
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
<configuration>

<!-- Site specific YARN configuration properties -->

        <property>
                  <name>yarn.resourcemanager.webapp.address</name>
                            <value>master:8088</value>
                    </property>
                    <property>
                              <name>yarn.resourcemanager.hostname</name>
                                        <value>master</value>
                                </property>
                                <property>
                                          <name>yarn.nodemanager.aux-services</name>
                                                    <value>mapreduce_shuffle</value>
                                            </property>
</configuration>
~                                                                                                                       ~                                                                                                                       ~                                                                                                                       ~                                                                                                                       ~                                                                                                                       ~        
```



workers

```
master
slave1
slave2
~                                                                                                                       ~                                                                                                                       ~                                                                                                                       ~                                                                                                                       ~         
```

在$HADOOP_HOME目录下创建datanode,namenode,tmp文件夹





HBASE



hbase-env.sh

```
export HBASE_DISABLE_HADOOP_CLASSPATH_LOOKUP="true"

# Override text processing tools for use by these launch scripts.
# export GREP="${GREP-grep}"
# export SED="${SED-sed}"
export JAVA_HOME=/usr/local/jdk1.8.0_291
export HBASE_MANAGES_ZK=false
```



hbase-site.xml

```
* Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
-->
<configuration>
  <!--
    The following properties are set for running HBase as a single process on a
    developer workstation. With this configuration, HBase is running in
    "stand-alone" mode and without a distributed file system. In this mode, and
    without further configuration, HBase and ZooKeeper data are stored on the
    local filesystem, in a path under the value configured for `hbase.tmp.dir`.
    This value is overridden from its default value of `/tmp` because many
    systems clean `/tmp` on a regular basis. Instead, it points to a path within
    this HBase installation directory.

    Running against the `LocalFileSystem`, as opposed to a distributed
    filesystem, runs the risk of data integrity issues and data loss. Normally
    HBase will refuse to run in such an environment. Setting
    `hbase.unsafe.stream.capability.enforce` to `false` overrides this behavior,
    permitting operation. This configuration is for the developer workstation
    only and __should not be used in production!__

    See also https://hbase.apache.org/book.html#standalone_dist
  -->
         <property>
                         <name>hbase.rootdir</name>
                                         <value>hdfs://master:9000/hbase</value>
                                         <description>nothing</description>                              </property>
  <property>
    <name>hbase.cluster.distributed</name>
    <value>true</value>
  </property>
  <property>
                  <name>hbase.zookeeper.quorum</name>
                                  <value>master:2181,slave1:2181,slave2:2181</value>
                                                          <description>nothing</description>
                                                                      </property>
  <property>
    <name>hbase.tmp.dir</name>
    <value>./tmp</value>
  </property>
  <property>
    <name>hbase.unsafe.stream.capability.enforce</name>
    <value>false</value>
  </property>
</configuration>
```



regionservers

```
master
slave1
slave2
~                                                                                                                       ~         :q

```





HIVE

hive-site.xml

```
?xml version="1.0" encoding="UTF-8" standalone="no"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
	<property>   
    	<name>javax.jdo.option.ConnectionURL</name>                                 <value>jdbc:mysql://master:3306/hive?createDatabaseIfNotExist=true&amp;useUnicode=true&amp;characterEncoding=UTF-8&amp;useSSL=false</value>
    </property>                                                                 <property>
		<name>javax.jdo.option.ConnectionDriverName</name>
		<value>com.mysql.jdbc.Driver</value>
	</property>
	<property>
		<name>javax.jdo.option.ConnectionUserName</name>
		<value>root</value>
	</property>
	<property>
		<name>javax.jdo.option.ConnectionPassword</name>
		<value>asdfqwer</value>
	</property>
	<property>
		<name>datanucleus.readOnlyDatastore</name>
		<value>false</value>
	</property>                                                                 <property>
		<name>datanucleus.fixedDatastore</name>
		<value>false</value>
	</property>
    <property>
		<name>datanucleus.autoCreateSchema</name>
		<value>true</value>
	</property>
	<property>
		<name>datanucleus.schema.autoCreateAll</name>
		<value>true</value>
	</property>
	<property>
		<name>datanucleus.autoCreateTables</name>
		<value>true</value>
	</property>
	<property>
		<name>datanucleus.autoCreateColumns</name>
		<value>true</value>
	</property>
	<property>
		<name>hive.metastore.local</name>
		<value>true</value>
	</property>
	<property>
		<name>hive.cli.print.header</name>
		<value>true</value>
	</property>
	<property>
		<name>hive.cli.print.current.db</name>
        <value>true</value>
    </property>                                                             </configuration>                                                                                  
```



创建warehouse文件夹，将template去掉，

```
mysql-connector-java-5.1.49-bin.jar ##来自下载
guava-27.0-jre.jar##来自hadoop
```



SPARK



spark-env.sh

```
export JAVA_HOME=/usr/local/jdk1.8.0_291
export SCALA_HOME=/usr/share/scala
export HADOOP_HOME=/usr/local/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export SPARK_MASTER_HOST=master
export SPARK_LOCAL_DIRS=/usr/local/spark
```

workers

```
master
slave1
slave2
~            
```



```
start-spark-all.sh
stop-spark-all.sh
### 防止与Hadoop start-all.sh 冲突，改名
```



ZOOKEEPER



zoo.cfg

```
# The number of milliseconds of each tick
tickTime=2000
# The number of ticks that the initial
# synchronization phase can take
initLimit=10
# The number of ticks that can pass between
# sending a request and getting an acknowledgement
syncLimit=5
# the directory where the snapshot is stored.
# do not use /tmp for storage, /tmp here is just
# example sakes.
dataDir=/usr/local/zookeeper/data
# the port at which the clients will connect
clientPort=2181
server.1=172.17.0.2:2888:3888
server.2=172.17.0.3:2888:3888
server.3=172.17.0.4:2888:3888
```







~/.bashrc

```
export JAVA_HOME=/usr/local/jdk1.8.0_291
export PATH=$PATH:$JAVA_HOME/bin
export HADOOP_HOME=/usr/local/hadoop
export HADOOP_CONFIG_HOME=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
export PATH=$PATH:$HADOOP_HOME/sbin
export ZOOKEEPER_HOME=/usr/local/zookeeper
export PATH=$PATH:$ZOOKEEPER_HOME/bin
export SCALA_HOME=/usr/local/scala
export PATH=$PATH:$SCALA_HOME/bin
export SPARK_HOME=/usr/local/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
export HBASE_HOME=/usr/local/hbase
export PATH=$PATH:$HBASE_HOME/bin
export HIVE_HOME=/usr/local/hive
export PATH=$PATH:$HIVE_HOME/bin
```



# Hadoop



## step 1 拉取Ubuntu镜像

``` 
docker pull ubuntu:latest
```

## step 2 使用Dockerfile构建包含jdk的ubuntu镜像

```
去jdk官网下载jdk包，此处下载的为jdk1.8 ** jdk-8u291-linux-x64.tar.gz**, 将下载好的jdk文件移至wsl2环境下，在此目录下新建Dockerfile文件，并进入编辑状态
```

```
vim Dockerfile
```

在Dockfile中输入以下内容：

```
FROM ubuntu:latest
MAINTAINER duanmu
ADD jdk-8u291-linux-x64.tar.gz /usr/local/
ENV JAVA_HOME /usr/local/jdk1.8.0_291
ENV CLASSPATH $JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
ENV PATH $PATH:$JAVA_HOME/bin
```

编辑后保存，开始build镜像

```
docker build -t jdk-20210127 .
```

##  step 3 进入镜像 

​		新建一个以jdk-20210127为基础镜像的容器命名为ubuntu_hadoop并指定容器的hostname为charlie,并进入容器。

```
docker run -it --name=ubuntu_hadoop -h charlie jdk-20210127

```



## step 4 升级apt-get

```
apt-get update
```



## step 5 安装vim

```
apt-get install vim

```

## step 6 更新apt-get镜像源

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

## step 7 重新升级apt-get

```
apt-get update
```

## step 8 安装wget

```
apt-get update
```

## step 9 通过wget下载Hadoop安装包

```
wget https://mirrors.cnnic.cn/apache/hadoop/common/hadoop-3.2.2/hadoop-3.2.2.tar.gz

#wget https://mirrors.tuna.tsinghua.edu.cn/apache/hive/hive-3.1.2/apache-hive-3.1.2-bin.tar.gz

#wget https://mirrors.tuna.tsinghua.edu.cn/apache/zookeeper/zookeeper-3.6.3/apache-zookeeper-3.6.3-bin.tar.gz 

#wget https://mirrors.tuna.tsinghua.edu.cn/apache/hbase/stable/hbase-2.3.5-bin.tar.gz 

#wget https://mirrors.tuna.tsinghua.edu.cn/apache/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz

#wget https://downloads.lightbend.com/scala/2.13.6/scala-2.13.6.tgz

#wget http://mirrors.ustc.edu.cn/mysql-ftp/Downloads/MySQL-5.7/mysql-server_5.7.31-1ubuntu18.04_amd64.deb-bundle.tar

#wget https://cdn.mysql.com/archives/mysql-connector-java-5.1/mysql-connector-java-5.1.49.tar.gz
```

### step 10 解压hadoop

```
tar -xvzf hadoop-3.2.2.tar.gz -C /usr/local
```



### step 11 配置环境变量并重启配置文件

```
vim ~/.bashrc
```

新增以下环境变量：

```
export JAVA_HOME=/usr/local/jdk1.8.0_291
export PATH=$PATH:$JAVA_HOME/bin
export HADOOP_HOME=/usr/local/hadoop
export HADOOP_CONFIG_HOME=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
export PATH=$PATH:$HADOOP_HOME/sbin
```

并重启配置文件

```
source ~/.bashrc
```

### step 12 创建文件夹并修改配置文件

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
		<value>file:/usr/local/hadoop/tmp</value>
		<description>Abase for other temporary directories.</description>  		</property>
	<property>
		<name>fs.defaultFS</name>
		<value>hdfs://hdcluster</value>
	</property>
	<property>
		<name>ha.zookeeper.quorum</name>
		<value>master:2181,slave1:2181,slave2:2181</value>
	</property>
</configuration>
```

修改hdfs-site.xml

```
vim hdfs-site.xml
```

用下面配置替换

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
        <value>/usr/local/hadoop/namenode</value>
        <final>true</final>
	</property>
	<property>
        <name>dfs.datanode.name.dir</name>
        <value>/usr/local/hadoop/datanode</value>
        <final>true</final>
	</property>
	<property>
		<name>dfs.namenode.secondary.http-address</name>
		<value>master:50090</value>
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
		<name>mapreduce.framework.name</name>
		<value>yarn</value>
	</property>
    <property>
		<name>mapreduce.jobhistory.address</name>
		<value>master:10020</value>
	</property>
	<property>
		<name>mapreduce.jobhistory.webapp.address</name>
		<value>master:19888</value>
	</property>
</configuration>
```

再是yarn-site.xml

```
vim yarn-site.xml
```

使用下面的内容替换

```
<?xml version="1.0"?>
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
<configuration>

<!-- Site specific YARN configuration properties -->
	<property>
		<name>yarn.resourcemanager.webapp.address</name>
		<value>master:8088</value>
	</property>
	<property>
		<name>yarn.resourcemanager.hostname</name>
		<value>master</value>
	</property>
	<property>
		<name>yarn.nodemanager.aux-services</name>
		<value>mapreduce_shuffle</value>
	</property>
</configuration>
```

修改hadoop环境变量，在hadoop安装目录下，找到hadoop-env.sh文件

```
vim hadoop-env.sh
```

在最后添加

```
export JAVA_HOME=/usr/local/jdk1.8.0_291
export HDFS_NAMENODE_USER=root
export HDFS_DATANODE_USER=root
export HDFS_SECONDARYNAMENODE_USER=root
export YARN_RESOURCEMANAGER_USER=root
export YARN_NODEMANAGER_USER=root
```

编辑安装目录下的workers文件

```
vim workers
```

内容改为

```
master
slave1
slave2
```

### 刷新及hdfs初始化

```

chown -R root:root /usr/local/hadoop/
```

## 安装SSH

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

## 修改SSH配置

```
vim /etc/ssh/ssh_config
```

添加,将下面这句话直接添加即可，也可以在文件中找到被注释的这句话去修改。

```
StrictHostKeyChecking no #将ask改为no
```

```
vim /etc/ssh/sshd_config
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

用户组问题，chown -R  root:root .ssh

权限问题 

```
chmod g-w /home/your_user # 或　chmod 0755 /home/your_user
 
chmod 700 /home/your_user/.ssh
 
chmod 600 /home/your_user/.ssh/authorized_keys
chmod 600 .ssh/ida_*

```

将hadoop文件夹利用scp传输

```
scp -r /usr/local/hadoop/ slave1:/usr/local/
scp -r /usr/local/hadoop/ slave2:/usr/local/
```



在master机器上初始化

```
hdfs namenode -format #否则web端看不到
```

# Zookeeper安装配置

```
wget https://mirrors.tuna.tsinghua.edu.cn/apache/zookeeper/zookeeper-3.6.3/apache-zookeeper-3.6.3-bin.tar.gz
#下载完成后解压至/usr/local目录下
tar -zxvf apache-zookeeper-3.6.1-bin.tar.gz -C /usr/local/
cd /usr/local
# 重命名zookeeper
mv apache-zookeeper-3.6.1-bin zookeeper
```

设置环境变量

```
vim ~/.bashrc
```

添加

```
export ZOOKEEPER_HOME=/usr/local/zookeeper
export PATH=$PATH:$ZOOKEEPER_HOME/bin
```

分发至其他机器并执行

```
source ~/.bashrc
```

配置zookeeper

进入conf目录

```
cd /usr/local/zookeeper/conf
```

将zoo_sample.cfg复制一份并命名为zoo.cfg

```
cp zoo_sample.cfg zoo.cfg
```

对zoo.cfg做如下修改

```
dataDir=/usr/local/zookeeper/data
server.1=master:2888:3888
server.2=slave1:2888:3888
server.3=slave2:2888:3888
```



分发至其他机器

创建data目录并新建一个myid 的文件，在每个机器中，文件内容对应server.后面的数字，master机器即为

```
vim /usr/local/zookeeper/data/myid

插入1
```

至此还只是集群搭建，但不是高可用！

# Spark 安装配置



## spark

解压spark文件

```
tar -xvf spark-3.1.2-bin-hadoop3.2.tgz -C /usr/local
# 然后重命名
cd /usr/local
mv spark-3.1.2-bin-hadoop3.2 spark
```

环境配置

```
vim ~/.bashrc
```

更改配置文件

```
cd /usr/local/spark/conf
vi spark-env.sh
```

写入以下文件

```
export JAVA_HOME=/usr/local/jdk1.8.0_291
export SCALA_HOME=/usr/share/scala
export HADOOP_HOME=/usr/local/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export SPARK_MASTER_HOST=master
export SPARK_LOCAL_DIRS=/usr/local/spark
```

再同一目录下

```
vim workers
```

写入以下内容

```
master
slave1
slave2
```

将/usr/local/spark/sbin目录下start-all.sh 重命名为start-spark-all.sh, stop-all.sh 重命名为stop-spark-all.sh



## scala 安装配置

只需在~/.bashrc

添加

```
export SCALA_HOME=/usr/local/scala
export PATH=$PATH:$SCALA_HOME/bin
```

# Hbase 安装配置

解压下载文件

```
tar -xvf hbase-2.3.5-bin.tar.gz -C /usr/local/
#重命名
cd /usr/local
 mv hbase-2.3.5/ hbase
export HBASE_DISABLE_HADOOP_CLASSPATH_LOOKUP="true"
export JAVA_HOME=/usr/local/jdk1.8.0_291
export HBASE_MANAGES_ZK=false
```

修改~/.bashrc

```
export HBASE_HOME=/usr/local/hbase
export PATH=$PATH:$HBASE_HOME/bin
```

修改配置文件

```
cd /usr/local/hbase/conf
vi hbase-env.sh
```

添加以下内容

```
export HBASE_DISABLE_HADOOP_CLASSPATH_LOOKUP="true"
export JAVA_HOME=/usr/local/jdk1.8.0_291
export HBASE_MANAGES_ZK=false
```

修改hbase-site.xml

```
  <property>
                         <name>hbase.rootdir</name>
                                         <value>hdfs://master:9000/hbase</value>
                                         <description>nothing</description>                              </property>
  <property>
    <name>hbase.cluster.distributed</name>
    <value>true</value>
  </property>
  <property>
                  <name>hbase.zookeeper.quorum</name>
                                  <value>master:2181,slave1:2181,slave2:2181</value>
                                                          <description>nothing</description>
                                                                      </property>
  <property>
    <name>hbase.tmp.dir</name>
    <value>./tmp</value>
  </property>
  <property>
    <name>hbase.unsafe.stream.capability.enforce</name>
    <value>false</value>
  </property>
```



修改regionservers

```
master
slave1
slave2
```

# Hive安装配置



## mysql安装

```
tar -xvf mysql-server_5.7.31-1ubuntu18.04_amd64.deb-bundle.tar
apt-get install ./libmysql*
apt-get install libtinfo5
apt-get install ./mysql-community-client_5.7.31-1ubuntu18.04_amd64.deb
apt-get install ./mysql-client_5.7.31-1ubuntu18.04_amd64.deb
apt-get install ./mysql-community-server_5.7.31-1ubuntu18.04_amd64.deb
###第6行步骤会有两次让输入密码
apt-get install ./mysql-server_5.7.31-1ubuntu18.04_amd64.deb
###安装结束后，修改权限
cd /var/run
chmod -R 777 mysqld
cd /var/lib
chmod -R 777 mysql
service mysql start
mysql -uroot -p #输入密码


use mysql;
grant all privileges on *.* to 'hive'@'%' identified BY 'yourpassword' with grant option;
flush privileges;
exit;


service mysql restart


```

## hive 安装

解压

```
tar -xzvf apache-hive-3.1.2-bin.tar.gz -C /usr/local/
#重命名
cd /usr/local
mv apache-hive-3.1.2-bin hive
```

修改环境变量

```
vim ~/.bashrc
#添加以下内容
export HIVE_HOME=/usr/local/hive
export PATH=$PATH:$HIVE_HOME/bin
```

创建warehouse 文件夹

```
cd /usr/local/hive
mkdir warehouse
```

配置文件修改

hive-env.sh

```
HADOOP_HOME=/usr/local/hadoop
export HIVE_CONF_DIR=/usr/local/hive/conf
export HIVE_AUX_JARS_PATH=/usr/local/hive/lib
```

hive-site.xml

```
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
		<property>
					<name>javax.jdo.option.ConnectionURL</name>
	 <value>jdbc:mysql://master:3306/hive?createDatabaseIfNotExist=true&amp;useUnicode=true&amp;characterEncoding=UTF-8&amp;useSSL=false</value>
		 	</property>
					<property>
	<name>javax.jdo.option.ConnectionDriverName</name>
				<value>com.mysql.jdbc.Driver</value>
						</property>
	<property>
				<name>javax.jdo.option.ConnectionUserName</name>
<value>root</value>
		</property>
				<property>
<name>javax.jdo.option.ConnectionPassword</name>
<value>asdfqwer</value>
				</property>
<property>
			<name>datanucleus.readOnlyDatastore</name>
						<value>false</value>
	</property>
			<property>
						<name>datanucleus.fixedDatastore</name>
		<value>false</value>
				</property>
						<property>
		<name>datanucleus.autoCreateSchema</name>
					<value>true</value>
</property>
		<property>
					<name>datanucleus.schema.autoCreateAll</name>
	<value>true</value>
			</property>
					<property>
	<name>datanucleus.autoCreateTables</name>
				<value>true</value>
						</property>
	<property>
				<name>datanucleus.autoCreateColumns</name>
<value>true</value>
		</property>
				<property>
<name>hive.metastore.local</name>
			<value>true</value>
					</property>
<!-- 显示表的列名 -->
<property>
			<name>hive.cli.print.header</name>
						<value>true</value>
	</property>
			<!-- 显示数据库名称 -->
			<property>
						<name>hive.cli.print.current.db</name>
		<value>true</value>
				</property>
			</configuration>


```

客户端hive-site.xml[<sup>1</sup>](#refer-anchor-1)

```
<configuration>
        <property>
                  <name>hive.metastore.warehouse.dir</name>
                  <value>/usr/local/hive/warehouse</value>
                    </property>
                    <property>
                              <name>hive.metastore.local</name>
                                        <value>false</value>
                                </property>
                                <property>
                                          <name>hive.metastore.uris</name>
                                                    <value>thrift://master:9083</value>
                                            </property>
</configuration>
~           

```



```
#服务器端
schematool -dbType mysql -initSchema
hive --service metastore
客户端
hive
show databases;
```



第一次运行报错：

![image-20210719140401396](image-20210719140401396.png)





# 高可用

core-site.xml

```
<configuration>
	<property>
		<name>hadoop.tmp.dir</name>
		<value>file:/usr/local/hadoop/tmp</value>
		<description>Abase for other temporary directories.</description>  		</property>
	<property>
		<name>fs.defaultFS</name>
		<value>hdfs://hdcluster</value>
	</property>
	<property>
		<name>ha.zookeeper.quorum</name>
		<value>master:2181,slave1:2181,slave2:2181</value>
	</property>
</configuration>
```





hdfs-site.xml

```
<?xml version="1.0" encoding="UTF-8"?>
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
		<name>dfs.nameservices</name>
		<value>hdcluster</value>
	</property>
	<!-- myNameService1下面有两个NameNode，分别是nn1，nn2 -->
	<property>
		<name>dfs.ha.namenodes.hdcluster</name>
		<value>nn1,nn2</value>
	</property>
	<!-- nn1的RPC通信地址 -->
	 <property>
		<name>dfs.namenode.rpc-address.hdcluster.nn1</name>
		<value>master:9000</value>
	</property>
	<!-- nn1的http通信地址 -->
	<property>
		<name>dfs.namenode.http-address.hdcluster.nn1</name>
		<value>master:50070</value>
	</property>
	<!-- nn2的RPC通信地址 -->
	<property>
		<name>dfs.namenode.rpc-address.hdcluster.nn2</name>
		<value>slave1:9000</value>
	</property>
<!-- nn2的http通信地址 -->
	<property>
		<name>dfs.namenode.http-address.hdcluster.nn2</name>
		<value>slave1:50070</value>
	</property>
<!-- 指定NameNode的元数据在JournalNode上的存放位置 -->
	<property>
		<name>dfs.namenode.shared.edits.dir</name>
        <value>qjournal://master:8485;slave1:8485;slave2:8485/hdcluster</value>
	</property>
	<!-- 指定JournalNode在本地磁盘存放数据的位置 -->
	<property>
		<name>dfs.journalnode.edits.dir</name>
		<value>/usr/local/hadoop/journalData</value>
	</property>
<!-- 开启NameNode失败自动切换 -->
	<property>
		<name>dfs.ha.automatic-failover.enabled</name>
		<value>true</value>
	</property>
<!-- 配置失败自动切换实现方式 -->
	<property>
		<name>dfs.client.failover.proxy.provider.hdcluster</name>
		<value>org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider</value>
	</property>
<!-- 配置隔离机制方法，Failover后防止停掉的Namenode启动，造成两个服务,多个机制用换行分割，即每个机制暂用一行-->
    <property>
		<name>dfs.ha.fencing.methods</name>
		<value>sshfence
				shell(/bin/true)
		</value>
	</property>
<!-- 使用sshfence隔离机制时需要ssh免登陆，注意换成自己的用户名 -->
	<property>
		<name>dfs.ha.fencing.ssh.private-key-files</name>
		<value>/root/.ssh/id_rsa</value>
	</property>
	<property>
		<name>dfs.ha.fencing.ssh.connect-timeout</name>
		<value>30000</value>
	</property>
	<property>
		<name>dfs.permissions</name>
		<value>false</value>
	</property>
	<property>
		<name>dfs.replication</name>
		<value>2</value>
		<final>true</final>
	</property>
	<property>
		<name>dfs.namenode.name.dir</name>
		<value>/usr/local/hadoop/namenode</value>
		<final>true</final>
	</property>
	<property>
		<name>dfs.datanode.name.dir</name>
		<value>/usr/local/hadoop/datanode</value>
		<final>true</final>
	</property>
	<property>
		<name>dfs.namenode.secondary.http-address</name>
		<value>master:50090</value>
	</property>
</configuration>

```





hadoop-env.sh

```
export JAVA_HOME=/usr/local/jdk1.8.0_291
export HDFS_NAMENODE_USER=root
export HDFS_DATANODE_USER=root
export HDFS_SECONDARYNAMENODE_USER=root
export YARN_RESOURCEMANAGER_USER=root
export YARN_NODEMANAGER_USER=root
export HDFS_JOURNALNODE_USER=root
export HDFS_ZKFC_USER=root

```

mapred-site.xml

```
<configuration>
	<property>
		<name>mapreduce.framework.name</name>
		<value>yarn</value>
	</property>
	<property>
		<name>mapreduce.jobhistory.address</name>
		<value>master:10020</value>
	</property>
	<property>
		<name>mapreduce.jobhistory.webapp.address</name>
		<value>master:19888</value>
	</property>
</configuration>

```

yarn-site.xml

```
<configuration>
	<property>
		<name>yarn.resourcemanager.ha.enabled</name>
		<value>true</value>
	</property>
	<property>
		<name>yarn.resourcemanager.cluster-id</name>
		<value>yrc</value>
	</property>
	<property>
		<name>yarn.resourcemanager.ha.rm-ids</name>
 		<value>rm1,rm2</value>
	</property>
	<property>
		<name>yarn.resourcemanager.hostname.rm1</name>
		<value>master</value>
	</property>
	<property>
		<name>yarn.resourcemanager.hostname.rm2</name>
		<value>slave1</value>
	</property>
	<property>
		<name>yarn.resourcemanager.webapp.address.rm1</name>
	    <value>master:8088</value>
	</property>
	<property>
		<name>yarn.resourcemanager.webapp.address.rm2</name>		
		<value>slave1:8088</value>
    </property>																	<property>
		<name>yarn.resourcemanager.zk-address</name>
		<value>master:2181,slave1:2181</value>
	</property>
	<property>
		<name>yarn.nodemanager.aux-services</name>
		<value>mapreduce_shuffle</value>
	</property>
	<property>
		<name>yarn.application.classpath</name>
					<value>/usr/local/hadoop/etc/hadoop:/usr/local/hadoop/share/hadoop/common/lib/*:/usr/local/hadoop/share/hadoop/common/*:/usr/local/hadoop/share/hadoop/hdfs:/usr/local/hadoop/share/hadoop/hdfs/lib/*:/usr/local/hadoop/share/hadoop/hdfs/*:/usr/local/hadoop/share/hadoop/mapreduce/lib/*:/usr/local/hadoop/share/hadoop/mapreduce/*:/usr/local/hadoop/share/hadoop/yarn:/usr/local/hadoop/share/hadoop/yarn/lib/*:/usr/local/hadoop/share/hadoop/yarn/*</value>
	</property>
</configuration>

```





**运行错误**：

**ERROR: Cannot set priority of datanode process**

**解决方案**

​	**chown -R root:root ##权限问题**

​	**也有可能是配置文件不一致**

如果初始化过，且journaldata可能初始化过，再次初始化namenode,则需要先启动journalnode

```
错误提示：
Unable to check if JNs are ready for formatting. 
```



```
hadoop-daemon.sh start journalnode->hdfs --daemon start journalnode
```



```
<configuration>
	<property>
		<name>dfs.replication</name>
		<value>2</value>
	</property>
	<property>
		<name>dfs.nameservices</name>
		<value>hdcluster</value>
	</property>
	<property>
		<name>dfs.ha.namenodes.hdcluster</name>
		<value>nn1,nn2</value>
	</property>
	<property>
		<name>dfs.namenode.rpc-address.hdcluster.nn1</name>
		<value>master:8020</value>
	</property>
	<property>
		<name>dfs.namenode.rpc-address.hdcluster.nn2</name>
		<value>slave1:8020</value>
	</property>
	<property>
		<name>dfs.namenode.http-address.hdcluster.nn1</name>
		<value>master:9870</value>
	</property>
	<property>
		<name>dfs.namenode.http-address.hdcluster.nn2</name>
		<value>slave1:9870</value>
	</property>
	<property>
		<name>dfs.namenode.shared.edits.dir</name>
		<value>qjournal://master:8485;slave1:8485;slave2:8485/hdcluster</value>
	</property>
	<property>
		<name>dfs.client.failover.proxy.provider.hdcluster</name>
	    <value>org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider</value>
	</property>
	<property>
		<name>dfs.ha.fencing.methods</name>
		<value>sshfence</value>
	</property>
	<property>
		<name>dfs.ha.fencing.ssh.private-key-files</name>
		<value>/root/.ssh/id_rsa</value>
	</property>
	<property>
		<name>dfs.journalnode.edits.dir</name>
		<value>/usr/local/hadoop/journalData</value>
	</property>
	<property>
		<name>dfs.ha.automatic-failover.enabled.hdcluster</name>
		<value>true</value>
	</property>
	<property>
        <name>dfs.namenode.name.dir</name>
        <value>/usr/local/hadoop/namenode</value>
        <final>true</final>
	</property>
	<property>
        <name>dfs.datanode.name.dir</name>
        <value>/usr/local/hadoop/datanode</value>
        <final>true</final>
	</property>
</configuration>
```



# Reference



<!--
&emsp;&emsp;<font face="黑体" size=10>16. 我是黑体字</font>  <div id="refer-anchor-1"></div>- [1] [hive搭建](https://www.jianshu.com/p/fd73c53668f5)
-->

<div style="display:none">这是一段注释</div>

1.  [ Docker配置Hadoop集群并使用WordCount测试_出大问题-CSDN博客](https://blog.csdn.net/weixin_43993764/article/details/113405025)
2.  [Hadoop3.2.1 HA 高可用集群的搭建（基于Zookeeper，NameNode高可用+Yarn高可用）_Captain.Y.的博客-CSDN博客](https://blog.csdn.net/weixin_43311978/article/details/106099052)
3.  [ CentOS7使用Docker安装hadoop集群_Captain.Y.的博客-CSDN博客](https://blog.csdn.net/weixin_43311978/article/details/105400694?spm=1001.2014.3001.5501)
4.  [Ubuntu下"sshd:unrecognized service"_子建莫敌-CSDN博客](https://blog.csdn.net/u013015629/article/details/70045809)
5.  [ Hadoop3.1.3+Zookeeper3.5.6+hbase2.2.2+hive3.1.2安装以及测试_井鱼的博客-CSDN博客](https://blog.csdn.net/weixin_43487121/article/details/103589532)
6.  [ Hadoop3.2 +Spark3.0全分布式安装_piaoxi6587的博客-CSDN博客](https://blog.csdn.net/piaoxi6587/article/details/103569376)
7.  [ 使用Paralles Desktop，在虚拟机环境中搭建hadoop集群（2主3从5节点）_shanhai3000的博客-CSDN博客](https://blog.csdn.net/shanhai3000/article/details/104865652)
8.  [安装并配置HBase集群（5个节点）_shanhai3000的博客-CSDN博客](https://blog.csdn.net/shanhai3000/article/details/107682499)
9.  [Docker 使用Dockerfile构建MySQL镜像（十五） - 勤奋的冬枣 (zabbx.cn)](https://www.zabbx.cn/archives/docker使用dockerfile构建mysql镜像十五)
10.  [(22条消息) Hexo-Next 主题博客个性化配置超详细，超全面(两万字)_Z小旋-CSDN博客_hexo next主题配置](https://blog.csdn.net/as480133937/article/details/100138838)
11.  [3.Spark环境搭建-Spark完全分布式集群搭建 - 简书 (jianshu.com)](https://www.jianshu.com/p/30d45fa044a2)
12.  [Apache Hadoop HDFS高可用部署实战案例 - JasonYin2020 - 博客园 (cnblogs.com)](https://www.cnblogs.com/yinzhengjie2020/p/12508145.html)
13.  [HBase的完全分布式搭建 - coder、 - 博客园 (cnblogs.com)](https://www.cnblogs.com/rmxd/p/11316062.html#_label4_0)
14.  [HBase完全分布式集群搭建 - JasonYin2020 - 博客园 (cnblogs.com)](https://www.cnblogs.com/yinzhengjie2020/p/12239031.html)
15.  [ hive安装及mysql配置_炼剑-CSDN博客_hive配置mysql](https://blog.csdn.net/agent_x/article/details/78660341)
16.  [ Mysql 8.0.13 开启远程访问权限（ERROR 1064 (42000): You have an error in your SQL syntax; check the manual th）_GentleCP的博客-CSDN博客](https://blog.csdn.net/GentleCP/article/details/87936263)
17.  [运行Spark-shell，解决Unable to load native-hadoop library for your platform - 提君 - 博客园 (cnblogs.com)](https://www.cnblogs.com/tijun/p/7562282.html)
18.  [ubuntu下hadoop、spark、hive、azkaban 集群搭建 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/89472385)
19.  [我的随笔 - 虚无境 - 博客园 (cnblogs.com)-大数据学习系列 ](https://www.cnblogs.com/xuwujing/p/)
20.  [Spark on Hive & Hive on Spark，傻傻分不清楚 - 云+社区 - 腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1624245)
21.  [hive on spark安装(hive2.3 spark2.1)_敲码的汉子-CSDN博客_hive on spark 安装](https://blog.csdn.net/Dante_003/article/details/72867493)
22.  [Hadoop Hive集群搭建（含CentOS和Ubuntu） - 随笔分类 - 大数据和AI躺过的坑 - 博客园 (cnblogs.com)](https://www.cnblogs.com/zlslch/category/965666.html)

23. [(22条消息) ssh公钥都追加到authorized_keys文件了，但是还是无法免秘钥登陆_孑然一身踽踽而行-CSDN博客](https://blog.csdn.net/u011809553/article/details/80937624)
24. [Hive学习笔记一：远程服务器模式搭建 - 简书 (jianshu.com)](https://www.jianshu.com/p/fd73c53668f5)

