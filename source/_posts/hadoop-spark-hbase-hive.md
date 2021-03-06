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

???$HADOOP_HOME???????????????datanode,namenode,tmp?????????





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



??????warehouse???????????????template?????????

```
mysql-connector-java-5.1.49-bin.jar ##????????????
guava-27.0-jre.jar##??????hadoop
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
### ?????????Hadoop start-all.sh ???????????????
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



## step 1 ??????Ubuntu??????

``` 
docker pull ubuntu:latest
```

## step 2 ??????Dockerfile????????????jdk???ubuntu??????

```
???jdk????????????jdk????????????????????????jdk1.8 ** jdk-8u291-linux-x64.tar.gz**, ???????????????jdk????????????wsl2?????????????????????????????????Dockerfile??????????????????????????????
```

```
vim Dockerfile
```

???Dockfile????????????????????????

```
FROM ubuntu:latest
MAINTAINER duanmu
ADD jdk-8u291-linux-x64.tar.gz /usr/local/
ENV JAVA_HOME /usr/local/jdk1.8.0_291
ENV CLASSPATH $JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
ENV PATH $PATH:$JAVA_HOME/bin
```

????????????????????????build??????

```
docker build -t jdk-20210127 .
```

##  step 3 ???????????? 

???		???????????????jdk-20210127?????????????????????????????????ubuntu_hadoop??????????????????hostname???charlie,??????????????????

```
docker run -it --name=ubuntu_hadoop -h charlie jdk-20210127

```



## step 4 ??????apt-get

```
apt-get update
```



## step 5 ??????vim

```
apt-get install vim

```

## step 6 ??????apt-get?????????

```
vim /etc/apt/sources.list
```

??????????????????????????????

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

## step 7 ????????????apt-get

```
apt-get update
```

## step 8 ??????wget

```
apt-get update
```

## step 9 ??????wget??????Hadoop?????????

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

### step 10 ??????hadoop

```
tar -xvzf hadoop-3.2.2.tar.gz -C /usr/local
```



### step 11 ???????????????????????????????????????

```
vim ~/.bashrc
```

???????????????????????????

```
export JAVA_HOME=/usr/local/jdk1.8.0_291
export PATH=$PATH:$JAVA_HOME/bin
export HADOOP_HOME=/usr/local/hadoop
export HADOOP_CONFIG_HOME=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
export PATH=$PATH:$HADOOP_HOME/sbin
```

?????????????????????

```
source ~/.bashrc
```

### step 12 ????????????????????????????????????

```
cd $HADOOP_HOME
mkdir tmp
mkdir namenode
mkdir datanode
```

??????????????????

```
cd $HADOOP_CONFIG_HOME
vim core-site.xml
```

?????????????????????

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

??????hdfs-site.xml

```
vim hdfs-site.xml
```

?????????????????????

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



?????????

```
vim mapred-site.xml
```

????????????????????????

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

??????yarn-site.xml

```
vim yarn-site.xml
```

???????????????????????????

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

??????hadoop??????????????????hadoop????????????????????????hadoop-env.sh??????

```
vim hadoop-env.sh
```

???????????????

```
export JAVA_HOME=/usr/local/jdk1.8.0_291
export HDFS_NAMENODE_USER=root
export HDFS_DATANODE_USER=root
export HDFS_SECONDARYNAMENODE_USER=root
export YARN_RESOURCEMANAGER_USER=root
export YARN_NODEMANAGER_USER=root
```

????????????????????????workers??????

```
vim workers
```

????????????

```
master
slave1
slave2
```

### ?????????hdfs?????????

```

chown -R root:root /usr/local/hadoop/
```

## ??????SSH

hadoop?????????????????????ssh????????????????????????ssh

```
apt-get install net-tools
apt-get install ssh
```

??????sshd??????

```
mkdir -p ~/var/run/sshd
```

??????????????????

```
cd ~/
ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
cd .ssh
cat id_rsa.pub >> authorized_keys
```

?????????????????????????????????????????????????????????????????????????????????????????????

## ??????SSH??????

```
vim /etc/ssh/ssh_config
```

??????,???????????????????????????????????????????????????????????????????????????????????????????????????

```
StrictHostKeyChecking no #???ask??????no
```

```
vim /etc/ssh/sshd_config
```

??????????????????

```
#??????????????????
PasswordAuthentication no
#??????????????????
RSAAuthentication yes
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
```

???????????????????????????????????????????????????

```
ssh localhost
```

????????????????????????????????????

```
 /etc/init.d/ssh restart
```

??????????????????chown -R  root:root .ssh

???????????? 

```
chmod g-w /home/your_user # ??????chmod 0755 /home/your_user
 
chmod 700 /home/your_user/.ssh
 
chmod 600 /home/your_user/.ssh/authorized_keys
chmod 600 .ssh/ida_*

```

???hadoop???????????????scp??????

```
scp -r /usr/local/hadoop/ slave1:/usr/local/
scp -r /usr/local/hadoop/ slave2:/usr/local/
```



???master??????????????????

```
hdfs namenode -format #??????web????????????
```

# Zookeeper????????????

```
wget https://mirrors.tuna.tsinghua.edu.cn/apache/zookeeper/zookeeper-3.6.3/apache-zookeeper-3.6.3-bin.tar.gz
#????????????????????????/usr/local?????????
tar -zxvf apache-zookeeper-3.6.1-bin.tar.gz -C /usr/local/
cd /usr/local
# ?????????zookeeper
mv apache-zookeeper-3.6.1-bin zookeeper
```

??????????????????

```
vim ~/.bashrc
```

??????

```
export ZOOKEEPER_HOME=/usr/local/zookeeper
export PATH=$PATH:$ZOOKEEPER_HOME/bin
```

??????????????????????????????

```
source ~/.bashrc
```

??????zookeeper

??????conf??????

```
cd /usr/local/zookeeper/conf
```

???zoo_sample.cfg????????????????????????zoo.cfg

```
cp zoo_sample.cfg zoo.cfg
```

???zoo.cfg???????????????

```
dataDir=/usr/local/zookeeper/data
server.1=master:2888:3888
server.2=slave1:2888:3888
server.3=slave2:2888:3888
```



?????????????????????

??????data?????????????????????myid ???????????????????????????????????????????????????server.??????????????????master????????????

```
vim /usr/local/zookeeper/data/myid

??????1
```

???????????????????????????????????????????????????

# Spark ????????????



## spark

??????spark??????

```
tar -xvf spark-3.1.2-bin-hadoop3.2.tgz -C /usr/local
# ???????????????
cd /usr/local
mv spark-3.1.2-bin-hadoop3.2 spark
```

????????????

```
vim ~/.bashrc
```

??????????????????

```
cd /usr/local/spark/conf
vi spark-env.sh
```

??????????????????

```
export JAVA_HOME=/usr/local/jdk1.8.0_291
export SCALA_HOME=/usr/share/scala
export HADOOP_HOME=/usr/local/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export SPARK_MASTER_HOST=master
export SPARK_LOCAL_DIRS=/usr/local/spark
```

??????????????????

```
vim workers
```

??????????????????

```
master
slave1
slave2
```

???/usr/local/spark/sbin?????????start-all.sh ????????????start-spark-all.sh, stop-all.sh ????????????stop-spark-all.sh



## scala ????????????

?????????~/.bashrc

??????

```
export SCALA_HOME=/usr/local/scala
export PATH=$PATH:$SCALA_HOME/bin
```

# Hbase ????????????

??????????????????

```
tar -xvf hbase-2.3.5-bin.tar.gz -C /usr/local/
#?????????
cd /usr/local
 mv hbase-2.3.5/ hbase
export HBASE_DISABLE_HADOOP_CLASSPATH_LOOKUP="true"
export JAVA_HOME=/usr/local/jdk1.8.0_291
export HBASE_MANAGES_ZK=false
```

??????~/.bashrc

```
export HBASE_HOME=/usr/local/hbase
export PATH=$PATH:$HBASE_HOME/bin
```

??????????????????

```
cd /usr/local/hbase/conf
vi hbase-env.sh
```

??????????????????

```
export HBASE_DISABLE_HADOOP_CLASSPATH_LOOKUP="true"
export JAVA_HOME=/usr/local/jdk1.8.0_291
export HBASE_MANAGES_ZK=false
```

??????hbase-site.xml

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



??????regionservers

```
master
slave1
slave2
```

# Hive????????????



## mysql??????

```
tar -xvf mysql-server_5.7.31-1ubuntu18.04_amd64.deb-bundle.tar
apt-get install ./libmysql*
apt-get install libtinfo5
apt-get install ./mysql-community-client_5.7.31-1ubuntu18.04_amd64.deb
apt-get install ./mysql-client_5.7.31-1ubuntu18.04_amd64.deb
apt-get install ./mysql-community-server_5.7.31-1ubuntu18.04_amd64.deb
###???6????????????????????????????????????
apt-get install ./mysql-server_5.7.31-1ubuntu18.04_amd64.deb
###??????????????????????????????
cd /var/run
chmod -R 777 mysqld
cd /var/lib
chmod -R 777 mysql
service mysql start
mysql -uroot -p #????????????


use mysql;
grant all privileges on *.* to 'hive'@'%' identified BY 'yourpassword' with grant option;
flush privileges;
exit;


service mysql restart


```

## hive ??????

??????

```
tar -xzvf apache-hive-3.1.2-bin.tar.gz -C /usr/local/
#?????????
cd /usr/local
mv apache-hive-3.1.2-bin hive
```

??????????????????

```
vim ~/.bashrc
#??????????????????
export HIVE_HOME=/usr/local/hive
export PATH=$PATH:$HIVE_HOME/bin
```

??????warehouse ?????????

```
cd /usr/local/hive
mkdir warehouse
```

??????????????????

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
<!-- ?????????????????? -->
<property>
			<name>hive.cli.print.header</name>
						<value>true</value>
	</property>
			<!-- ????????????????????? -->
			<property>
						<name>hive.cli.print.current.db</name>
		<value>true</value>
				</property>
			</configuration>


```

?????????hive-site.xml[<sup>1</sup>](#refer-anchor-1)

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
#????????????
schematool -dbType mysql -initSchema
hive --service metastore
?????????
hive
show databases;
```



????????????????????????

![image-20210719140401396](image-20210719140401396.png)





# ?????????

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
	<!-- myNameService1???????????????NameNode????????????nn1???nn2 -->
	<property>
		<name>dfs.ha.namenodes.hdcluster</name>
		<value>nn1,nn2</value>
	</property>
	<!-- nn1???RPC???????????? -->
	 <property>
		<name>dfs.namenode.rpc-address.hdcluster.nn1</name>
		<value>master:9000</value>
	</property>
	<!-- nn1???http???????????? -->
	<property>
		<name>dfs.namenode.http-address.hdcluster.nn1</name>
		<value>master:50070</value>
	</property>
	<!-- nn2???RPC???????????? -->
	<property>
		<name>dfs.namenode.rpc-address.hdcluster.nn2</name>
		<value>slave1:9000</value>
	</property>
<!-- nn2???http???????????? -->
	<property>
		<name>dfs.namenode.http-address.hdcluster.nn2</name>
		<value>slave1:50070</value>
	</property>
<!-- ??????NameNode???????????????JournalNode?????????????????? -->
	<property>
		<name>dfs.namenode.shared.edits.dir</name>
        <value>qjournal://master:8485;slave1:8485;slave2:8485/hdcluster</value>
	</property>
	<!-- ??????JournalNode???????????????????????????????????? -->
	<property>
		<name>dfs.journalnode.edits.dir</name>
		<value>/usr/local/hadoop/journalData</value>
	</property>
<!-- ??????NameNode?????????????????? -->
	<property>
		<name>dfs.ha.automatic-failover.enabled</name>
		<value>true</value>
	</property>
<!-- ???????????????????????????????????? -->
	<property>
		<name>dfs.client.failover.proxy.provider.hdcluster</name>
		<value>org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider</value>
	</property>
<!-- ???????????????????????????Failover??????????????????Namenode???????????????????????????,?????????????????????????????????????????????????????????-->
    <property>
		<name>dfs.ha.fencing.methods</name>
		<value>sshfence
				shell(/bin/true)
		</value>
	</property>
<!-- ??????sshfence?????????????????????ssh?????????????????????????????????????????? -->
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





**????????????**???

**ERROR: Cannot set priority of datanode process**

**????????????**

???	**chown -R root:root ##????????????**

???	**????????????????????????????????????**

????????????????????????journaldata????????????????????????????????????namenode,??????????????????journalnode

```
???????????????
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
&emsp;&emsp;<font face="??????" size=10>16. ???????????????</font>  <div id="refer-anchor-1"></div>- [1] [hive??????](https://www.jianshu.com/p/fd73c53668f5)
-->

<div style="display:none">??????????????????</div>

1.  [ Docker??????Hadoop???????????????WordCount??????_????????????-CSDN??????](https://blog.csdn.net/weixin_43993764/article/details/113405025)
2.  [Hadoop3.2.1 HA ?????????????????????????????????Zookeeper???NameNode?????????+Yarn????????????_Captain.Y.?????????-CSDN??????](https://blog.csdn.net/weixin_43311978/article/details/106099052)
3.  [ CentOS7??????Docker??????hadoop??????_Captain.Y.?????????-CSDN??????](https://blog.csdn.net/weixin_43311978/article/details/105400694?spm=1001.2014.3001.5501)
4.  [Ubuntu???"sshd:unrecognized service"_????????????-CSDN??????](https://blog.csdn.net/u013015629/article/details/70045809)
5.  [ Hadoop3.1.3+Zookeeper3.5.6+hbase2.2.2+hive3.1.2??????????????????_???????????????-CSDN??????](https://blog.csdn.net/weixin_43487121/article/details/103589532)
6.  [ Hadoop3.2 +Spark3.0??????????????????_piaoxi6587?????????-CSDN??????](https://blog.csdn.net/piaoxi6587/article/details/103569376)
7.  [ ??????Paralles Desktop??????????????????????????????hadoop?????????2???3???5?????????_shanhai3000?????????-CSDN??????](https://blog.csdn.net/shanhai3000/article/details/104865652)
8.  [???????????????HBase?????????5????????????_shanhai3000?????????-CSDN??????](https://blog.csdn.net/shanhai3000/article/details/107682499)
9.  [Docker ??????Dockerfile??????MySQL?????????????????? - ??????????????? (zabbx.cn)](https://www.zabbx.cn/archives/docker??????dockerfile??????mysql????????????)
10.  [(22?????????) Hexo-Next ????????????????????????????????????????????????(?????????)_Z??????-CSDN??????_hexo next????????????](https://blog.csdn.net/as480133937/article/details/100138838)
11.  [3.Spark????????????-Spark??????????????????????????? - ?????? (jianshu.com)](https://www.jianshu.com/p/30d45fa044a2)
12.  [Apache Hadoop HDFS??????????????????????????? - JasonYin2020 - ????????? (cnblogs.com)](https://www.cnblogs.com/yinzhengjie2020/p/12508145.html)
13.  [HBase???????????????????????? - coder??? - ????????? (cnblogs.com)](https://www.cnblogs.com/rmxd/p/11316062.html#_label4_0)
14.  [HBase??????????????????????????? - JasonYin2020 - ????????? (cnblogs.com)](https://www.cnblogs.com/yinzhengjie2020/p/12239031.html)
15.  [ hive?????????mysql??????_??????-CSDN??????_hive??????mysql](https://blog.csdn.net/agent_x/article/details/78660341)
16.  [ Mysql 8.0.13 ???????????????????????????ERROR 1064 (42000): You have an error in your SQL syntax; check the manual th???_GentleCP?????????-CSDN??????](https://blog.csdn.net/GentleCP/article/details/87936263)
17.  [??????Spark-shell?????????Unable to load native-hadoop library for your platform - ?????? - ????????? (cnblogs.com)](https://www.cnblogs.com/tijun/p/7562282.html)
18.  [ubuntu???hadoop???spark???hive???azkaban ???????????? - ?????? (zhihu.com)](https://zhuanlan.zhihu.com/p/89472385)
19.  [???????????? - ????????? - ????????? (cnblogs.com)-????????????????????? ](https://www.cnblogs.com/xuwujing/p/)
20.  [Spark on Hive & Hive on Spark????????????????????? - ???+?????? - ????????? (tencent.com)](https://cloud.tencent.com/developer/article/1624245)
21.  [hive on spark??????(hive2.3 spark2.1)_???????????????-CSDN??????_hive on spark ??????](https://blog.csdn.net/Dante_003/article/details/72867493)
22.  [Hadoop Hive??????????????????CentOS???Ubuntu??? - ???????????? - ????????????AI???????????? - ????????? (cnblogs.com)](https://www.cnblogs.com/zlslch/category/965666.html)

23. [(22?????????) ssh??????????????????authorized_keys?????????????????????????????????????????????_????????????????????????-CSDN??????](https://blog.csdn.net/u011809553/article/details/80937624)
24. [Hive????????????????????????????????????????????? - ?????? (jianshu.com)](https://www.jianshu.com/p/fd73c53668f5)

