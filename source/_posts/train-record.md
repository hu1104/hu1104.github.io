---
title: train_record
date: 2021-07-27 11:04:44
tags:
  - 加密
categories:
password: Mike
abstract: Welcome to my blog, enter password to read.
message: Welcome to my blog, enter password to read.
typora-root-url: ./
---

# 20210728



将backbone换成mobilenetv2[<sup>1</sup>](#id=1)

<div id="1"></div>

- [1]  [修改结构](https://blog.csdn.net/wa1tzy/article/details/114492726)



faster-rcnn 尝试失败，mmdection报错，待后续解决

问题：

w和h加错， x->row, y->c



# 20210810



##WBF 

尝试在detect出来的640*640图片上使用，但没有起作用，可能是检测出的目标太少



# 20210811

[mmdection](https://blog.csdn.net/qq_41251963/article/details/112940253)

myself/faster_rcnn_r50_fpn_1x_coco.py 修改data的ann_file配置

dos2unix windows传至服务器换行符啥的有可能出错



faster-rcnn 利用epoch_5.pth inference ， 结果保存在frcnn_result



# 20210824

yolox :

![image-20210824090625419](train-record/image-20210824090625419.png)

![image-20210824090430512](train-record/image-20210824090430512.png)



[修改yolox文件](https://zhuanlan.zhihu.com/p/397499216)



![image-20210825092352104](train-record/image-20210825092352104.png)





yolov5 新数据

整体结果：

![image-20210825174105987](train-record/image-20210825174105987.png)



![image-20210825172719804](train-record/image-20210825172719804.png)

疑似多标





![image-20210825172949734](train-record/image-20210825172949734.png)

这种疑似杂质标为单个

![image-20210825173901708](train-record/image-20210825173901708.png)

疑似虚影影响了





300轮

![image-20210826101712429](train-record/image-20210826101712429.png)

![image-20210826101507224](train-record/image-20210826101507224.png)

这里无论在100轮还是300轮都被识别成3个single，第一段+第二段、第三段、三段全





![image-20210826102001256](train-record/image-20210826102001256.png)

左边100轮，右边300轮

![image-20210826102122674](train-record/image-20210826102122674.png)

![image-20210826102134535](train-record/image-20210826102134535.png)

原图可能是single，两种都标为杂质

![image-20210826102316376](train-record/image-20210826102316376.png)

![image-20210826102457903](train-record/image-20210826102457903.png)

![image-20210826103640170](train-record/image-20210826103640170.png)

![image-20210826104054269](train-record/image-20210826104054269.png)

![image-20210826104044401](train-record/image-20210826104044401.png)

![image-20210826104835757](train-record/image-20210826104835757.png)

![image-20210826105000956](train-record/image-20210826105000956.png)

![image-20210826105011279](train-record/image-20210826105011279.png)

![image-20210826105034052](train-record/image-20210826105034052.png)

更倾向于右边

![image-20210826105705038](train-record/image-20210826105705038.png)

![image-20210826105714639](train-record/image-20210826105714639.png)

右下角single分段出现问题

![image-20210826110207298](train-record/image-20210826110207298.png)

![image-20210826110219710](train-record/image-20210826110219710.png)

![image-20210826110249288](train-record/image-20210826110249288.png)

![image-20210826110302359](train-record/image-20210826110302359.png)

![image-20210826110359832](train-record/image-20210826110359832.png)

![image-20210826110658790](train-record/image-20210826110658790.png)

![image-20210826110707014](train-record/image-20210826110707014.png)

![image-20210826110841652](train-record/image-20210826110841652.png)

![image-20210826110908397](train-record/image-20210826110908397.png)

左边为之前数据集，右边为原数据

![image-20210826111028100](train-record/image-20210826111028100.png)

![d](train-record/image-20210826111716177.png)

![image-20210826111737518](train-record/image-20210826111737518.png)

![image-20210826112053063](train-record/image-20210826112053063.png)

![image-20210826112111352](train-record/image-20210826112111352.png)

![image-20210826112207989](train-record/image-20210826112207989.png)

![image-20210826112221062](train-record/image-20210826112221062.png)

![image-20210826112558091](train-record/image-20210826112558091.png)

![image-20210826112609710](train-record/image-20210826112609710.png)

![image-20210826112718646](train-record/image-20210826112718646.png)

![image-20210826112738316](train-record/image-20210826112738316.png)

![image-20210826113008262](train-record/image-20210826113008262.png)



200轮

![image-20210827110136286](train-record/image-20210827110136286.png)





temp/4

677左右有部分浅的标签去掉没标



09-06标注进度![image-20210906180238487](train-record/image-20210906180238487.png)

 09-07 786张可能存在single，cluster都可以标的情况

![image-20210907142411310](/train-record/image-20210907142411310.png)



09-07标注进度![image-20210907175836344](/train-record/image-20210907175836344.png)



![image-20210908160847716](/train-record/image-20210908160847716.png)

![image-20210908160856393](/train-record/image-20210908160856393.png)





+4数据300轮

![image-20210909091914618](/train-record/image-20210909091914618.png)

![image-20210909103807895](train-record/image-20210909103807895.png)

+4数据100轮

![image-20210909091936485](./train-record/image-20210909091936485.png)

![image-20210909103824288](train-record/image-20210909103824288.png)

+4数据250轮

![image-20210909133753138](train-record/image-20210909133753138.png)

![image-20210909133808999](train-record/image-20210909133808999.png)

+4数据200轮

![image-20210909160339217](./train-record/image-20210909160339217.png)

![image-20210909160349528](./train-record/image-20210909160349528.png)





Mish 200轮

![image-20210910150001419](./train-record/image-20210910150001419.png)

![image-20210910153614275](./train-record/image-20210910153614275.png)



Mish 250轮

![image-20210910180227708](./train-record/image-20210910180227708.png)



CBAM 100

![image-20210915153848340](train-record/image-20210915153848340.png)



CBAM 200 

![image-20210916090937694](train-record/image-20210916090937694.png)



CBAM 300 

![image-20210916090959661](train-record/image-20210916090959661.png)



![img](train-record/企业微信截图_16317553364826.png)



CBAM300+1

CBAM300+300

![image-20210916161854393](train-record/image-20210916161854393.png)









# 坐标重新nms



717/1417 存在single和cluster融合

![image-20210910162226786](/train-record/image-20210910162226786.png)

715/1417 存在多个single可改为一个single

![image-20210910162317469](/train-record/image-20210910162317469.png)

705/1417  存在多个single可标为单个single

![image-20210910162414515](/train-record/image-20210910162414515.png)

698/1417 single和cluster融合

![image-20210910162935434](/train-record/image-20210910162935434.png)





新片记录

![image-20210930085715365](train-record/image-20210930085715365.png)

![image-20210930085748237](train-record/image-20210930085748237.png)

![image-20210930085902164](train-record/image-20210930085902164.png)

![image-20210930085932482](train-record/image-20210930085932482.png)

![image-20210930090019508](train-record/image-20210930090019508.png)

![image-20210930090054968](train-record/image-20210930090054968.png)

![image-20210930090106056](train-record/image-20210930090106056.png)

![image-20210930090306771](train-record/image-20210930090306771.png)

![image-20210930090326129](train-record/image-20210930090326129.png)

![image-20210930090400359](train-record/image-20210930090400359.png)

![image-20210930090445118](train-record/image-20210930090445118.png)

![image-20210930090455650](train-record/image-20210930090455650.png)

![image-20210930090549103](train-record/image-20210930090549103.png)

![image-20210930090638347](train-record/image-20210930090638347.png)

![image-20210930090649391](train-record/image-20210930090649391.png)

![image-20210930090713111](train-record/image-20210930090713111.png)

![image-20210930090723446](train-record/image-20210930090723446.png)

![image-20210930090741116](train-record/image-20210930090741116.png)

![image-20210930090750896](train-record/image-20210930090750896.png)

![image-20210930090817327](train-record/image-20210930090817327.png)

![image-20210930090854938](train-record/image-20210930090854938.png)

![image-20210930090916817](train-record/image-20210930090916817.png)

![image-20210930091008313](train-record/image-20210930091008313.png)

![image-20210930091148427](train-record/image-20210930091148427.png)

![image-20210930091533474](train-record/image-20210930091533474.png)

![image-20210930091544679](train-record/image-20210930091544679.png)

![image-20210930091604933](train-record/image-20210930091604933.png)

![image-20210930091653714](train-record/image-20210930091653714.png)

![image-20210930091704374](train-record/image-20210930091704374.png)

![image-20210930091745685](train-record/image-20210930091745685.png)

![image-20210930091937795](train-record/image-20210930091937795.png)

![image-20210930092256068](train-record/image-20210930092256068.png)

![image-20210930092452865](train-record/image-20210930092452865.png)

![image-20210930092507857](train-record/image-20210930092507857.png)

![image-20210930092533171](train-record/image-20210930092533171.png)

![image-20210930093056043](train-record/image-20210930093056043.png)

![image-20210930093107136](train-record/image-20210930093107136.png)

![image-20210930093117972](train-record/image-20210930093117972.png)

![image-20210930093913508](train-record/image-20210930093913508.png)

![image-20210930093949426](train-record/image-20210930093949426.png)

![image-20210930094041946](train-record/image-20210930094041946.png)

![image-20210930094050592](train-record/image-20210930094050592.png)

![image-20210930094105946](train-record/image-20210930094105946.png)

![image-20210930094218457](train-record/image-20210930094218457.png)

![image-20210930094234544](train-record/image-20210930094234544.png)

![image-20210930094250201](train-record/image-20210930094250201.png)

![image-20210930094609885](train-record/image-20210930094609885.png)

![image-20210930094629080](train-record/image-20210930094629080.png)

![image-20210930094735191](train-record/image-20210930094735191.png)

![image-20210930094822775](train-record/image-20210930094822775.png)

![image-20210930094830128](train-record/image-20210930094830128.png)

![image-20210930094840101](train-record/image-20210930094840101.png)

![image-20210930094851559](train-record/image-20210930094851559.png)

![image-20210930094859644](train-record/image-20210930094859644.png)

![image-20210930094909801](train-record/image-20210930094909801.png)

![image-20210930094920415](train-record/image-20210930094920415.png)

![image-20210930095033373](train-record/image-20210930095033373.png)

![image-20210930095057878](train-record/image-20210930095057878.png)

![image-20210930095113798](train-record/image-20210930095113798.png)

![image-20210930095151855](train-record/image-20210930095151855.png)

![image-20210930095159182](train-record/image-20210930095159182.png)

![image-20210930095236334](train-record/image-20210930095236334.png)

![image-20210930095718549](train-record/image-20210930095718549.png)

![image-20210930095735319](train-record/image-20210930095735319.png)

![image-20210930095745969](train-record/image-20210930095745969.png)

![image-20210930095805853](train-record/image-20210930095805853.png)

![image-20210930095821870](train-record/image-20210930095821870.png)

![image-20210930095835399](train-record/image-20210930095835399.png)

![image-20210930095900117](train-record/image-20210930095900117.png)

![image-20210930100013310](train-record/image-20210930100013310.png)

![image-20210930100138065](train-record/image-20210930100138065.png)

![image-20210930100325164](train-record/image-20210930100325164.png)

![image-20210930100334229](train-record/image-20210930100334229.png)

![image-20210930100343709](train-record/image-20210930100343709.png)

![image-20210930100422412](train-record/image-20210930100422412.png)

![image-20210930100453725](train-record/image-20210930100453725.png)

![image-20210930100502500](train-record/image-20210930100502500.png)

![image-20210930100530844](train-record/image-20210930100530844.png)

![image-20210930100557107](train-record/image-20210930100557107.png)

![image-20210930100606981](train-record/image-20210930100606981.png)

![image-20210930100616597](train-record/image-20210930100616597.png)





40倍

![image-20210930111825321](train-record/image-20210930111825321.png)