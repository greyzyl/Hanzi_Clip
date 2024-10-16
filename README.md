本代码为训练是添加cos_sim merge loss，仅使用部首笔画，测试时使用笔画、部首、笔画部首结合以及标准字库图图匹配和以上所有的ensemble

# install

参考你的https://github.com/FudanVI/FudanOCR/tree/main/image-ids-CTR

# 数据

1.训练及测试数据

通过网盘分享的文件：dataset.zip
链接: https://pan.baidu.com/s/195k5OVOUxtsrQz6-i-VJuA?pwd=yqvm 提取码: yqvm 
--来自百度网盘超级会员v7的分享

2.标准字库数据

通过网盘分享的文件：char_28762.zip
链接: https://pan.baidu.com/s/12B9XO7Cq8R36HNQMHZ2bMg?pwd=nc83 提取码: nc83 
--来自百度网盘超级会员v7的分享

# train

1、详情参考config文件

训练命令

```bash
python main_m.py
```

