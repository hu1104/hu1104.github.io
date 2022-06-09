---
title: theme
date: 2021-07-06 15:46:54
tags:
---



### 更改主题

进入命令行，下载 NexT 主题，输入:

```git clone https://github.com/theme-next/hexo-theme-next themes/next```

修改站点配置文件`_config.yml`，找到如下代码：

```
## Themes: https://hexo.io/themes/
theme: landscape
```

将 landscape 修改为 next 即可。

<!--more-->

### 隐藏网页底部 powered By Hexo / 强力驱动

打开 themes/next/layout/_partials/footer.swig

找到：

```
{% if theme.footer.powered.enable %}
<div class="powered-by">{#
#}{{ __('footer.powered', '<a class="theme-link" target="_blank"' + nofollow + ' href="https://hexo.io">Hexo</a>') }}{% if theme.footer.powered.version %} v{{ hexo_env('version') }}{% endif %}{#
#}</div>
{% endif %}
{% if theme.footer.powered.enable and theme.footer.theme.enable %}
<span class="post-meta-divider">|</span>
{% endif %}
{% if theme.footer.theme.enable %}
<div class="theme-info">{#
#}{{ __('footer.theme') }} – {#
#}<a class="theme-link" target="_blank"{{ nofollow }} href="https://theme-next.org">{#
#}NexT.{{ theme.scheme }}{#
#}</a>{% if theme.footer.theme.version %} v{{ version }}{% endif %}{#
#}</div>
{% endif %}
```

把这段代码首尾分别加上：`<!--` 和`-->`，或者直接删除。



### 统计字数和阅读时间



安装插件

```
npm install hexo-symbols-count-time --save
```

在站点配置文件_config.yml中添加如下代码：

```
symbols_count_time:
  symbols: true
  time: true
  total_symbols: true
  total_time: true
```



在主题配置文件找到对应代码修改为：

```
symbols_count_time:
  separated_meta: true
  item_text_post: true
  item_text_total: false
  awl: 4
  wpm: 275
```

最后：

```
hexo clean
hexo g 
hexo d
```





### 文档加密

安装插件：

```
npm install --save hexo-blog-encrypt
```

在站点配置文件中启用：

```
encrypt:
    enable: true
```

然后在文章头部加上对应字段password, abstract, message

```
---
title: 文章加密
date: 2019-01-04T22:20:13.000Z
category: 教程
tags:
  - 博客
  - Hexo
keywords: 博客文章密码
password: TloveY
abstract: 密码：TloveY
message:  输入密码，查看文章
---
```

```
password: 是该博客加密使用的密码
abstract: 是该博客的摘要，会显示在博客的列表页
message: 这个是博客查看时，密码输入框上面的描述性文字
```

### 部分显示

1. 写概述

   ```
   ---
   title: 让首页显示部分内容
   date: 2020-02-23 22:55:10
   description: 这是显示在首页的概述，正文内容均会被隐藏。
   ---
   ```

   

2. 文章截断

   在需要的地方插入

   ```
   <!--more-->
   ```

   
