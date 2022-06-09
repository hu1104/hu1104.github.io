---
title: logging
date: 2021-08-27 13:33:03
tags:
categories:
password:
abstract:
message:

---



# 按天生成日志

```
# #coding=utf-8
import logging,os  # 引入logging模块
from com_tools import setting


from logging import handlers

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射


    def __init__(self,filename,level='info',when='MIDNIGHT',backCount=7,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往控制台输出
        sh.setFormatter(format_str) #设置控制台上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,interval=1,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.suffix = "%Y-%m-%d.log" #设置文件后缀
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)
logfile = os.path.join(setting.logs_path, "daodianmockapi.txt") # 这个文件的名称就是当天的日志文件，过往的日志文件，会在后面追加文件后缀 th.suffix
logger = Logger(logfile,level='debug')

if __name__ == '__main__':
    #logger = Logger('all.log',level='debug')
    # filename = setting.now_time+ ".txt"
    # logfile = os.path.join(setting.logs_path,filename)
    # logger = Logger(logfile,level='debug')
    logger.logger.debug('debug') # 括号内的内容即为日志的文本内容
    logger.logger.info('info')
    logger.logger.warning('警告')
    logger.logger.error('报错')
    logger.logger.critical('严重')
    #Logger('error.log', level='error').logger.error('error')

```



# loguru





# 代码



```
import logging
import os
from logging.handlers import TimedRotatingFileHandler


class TRLogger:
    def __init__(self, logname, loglevel, logger):
        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        # 创建一个handler，用于写入日志文件
        if os.path.exists(os.path.join(os.getcwd(), logname)):
            os.system("rm -rf {}".format(os.path.join(os.getcwd(), logname)))

        fh = TimedRotatingFileHandler(logname, when='D', interval=1)
        fh.setLevel(logging.DEBUG)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        formatter = logging.Formatter('[%(asctime)s] - [logger name :%(name)s] - [%(filename)s file line:%(lineno)d] '
                                      '- %(levelname)s: %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def getlog(self):
        return self.logger



class Logger_:
    def __init__(self, logname, loglevel, logger):
        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        # 创建一个handler，用于写入日志文件
        if os.path.exists(os.path.join(os.getcwd(), logname)):
            os.system("rm -rf {}".format(os.path.join(os.getcwd(), logname)))


        fh = logging.FileHandler(logname)
        fh.setLevel(logging.DEBUG)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        formatter = logging.Formatter('[%(asctime)s] - [logger name :%(name)s] - [%(filename)s file line:%(lineno)d] '
                                      '- %(levelname)s: %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def getlog(self):
        return self.logger

```



# 参考资料



[python定时任务最强框架APScheduler详细教程 ](https://zhuanlan.zhihu.com/p/144506204)

[Loguru — 最强大的 Python 日志记录器](https://pythondict.com/life-intelligent/tools/loguru/)

[python 调试方法](https://blog.csdn.net/john_bh/article/details/107772357)

