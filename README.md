# PyTorch_YOLOv3
这个YOLOv3项目是配合我在知乎专栏上连载的《YOLO入门教程》而创建的：

https://zhuanlan.zhihu.com/c_1364967262269693952

感兴趣的小伙伴可以配合着上面的专栏来一起学习，入门目标检测。

另外，这个项目在小batch size 的情况，如batch size=8，可能会出现nan的问题，经过其他伙伴的调试，
在batch size=8时，可以把学习率lr跳到2e-4，兴许就可以稳定炼丹啦！ 我自己训练的时候，batch size
设置为16或32，比较大，所以训练稳定。

当然，这里也诚挚推荐我的另一个YOLO项目，训练更加稳定，性能更好呦

https://github.com/yjh0410/PyTorch_YOLO-Family

## DarkNet53预训练模型

大家可以从下面的百度网盘链接来下载已训练好的DarkNet53预模型：

链接: https://pan.baidu.com/s/1NmdqPwAmirknO5J__lg5Yw 

提起码: hlt6 

进入网盘中后，请点开网盘中的```backbone/darknet53/```文件夹，可以找到我们的DarkNet53预训练模型，包括```darknet53_75.42.pth```和```darknet53_hr_77.76.pth```文件，前者是在224图像上训练的，后者是经过448图像微调过的，即YOLOv2中的hi-res classifier训练技巧。

## 数据集

### VOC2007与VOC2012数据集

读者可以从下面的百度网盘链接来下载VOC2007和VOC2012数据集

链接：https://pan.baidu.com/s/1tYPGCYGyC0wjpC97H-zzMQ 

提取码：4la9

读者会获得 ```VOCdevkit.zip```压缩包, 分别包含 ```VOCdevkit/VOC2007``` 和 ```VOCdevkit/VOC2012```两个文件夹，分别是VOC2007数据集和VOC2012数据集.

### COCO 2017 数据集

运行 ```sh data/scripts/COCO2017.sh```，将会获得 COCO train2017, val2017, test2017三个数据集.

## 实验结果

官方的YOLOv3:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> 模型 </th>     <td bgcolor=white> 输入尺寸 </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3</th><td bgcolor=white> 320 </td><td bgcolor=white> 28.2 </td><td bgcolor=white> 51.5 </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3</th><td bgcolor=white> 416 </td><td bgcolor=white> 31.0 </td><td bgcolor=white> 55.3 </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3</th><td bgcolor=white> 608 </td><td bgcolor=white> 33.0 </td><td bgcolor=white> 57.0 </td><td bgcolor=white> 34.4 </td><td bgcolor=white> 18.3 </td><td bgcolor=white> 35.4 </td><td bgcolor=white> 41.9 </td></tr>
</table></tbody>

我们自己的 YOLOv3:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> 模型 </th>     <td bgcolor=white> 输入尺寸 </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3</th><td bgcolor=white> 320 </td><td bgcolor=white> 33.1 </td><td bgcolor=white> 54.1 </td><td bgcolor=white> 34.5 </td><td bgcolor=white> 12.1 </td><td bgcolor=white> 34.5 </td><td bgcolor=white> 49.6 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3</th><td bgcolor=white> 416 </td><td bgcolor=white> 36.0 </td><td bgcolor=white> 57.4 </td><td bgcolor=white> 37.0 </td><td bgcolor=white> 16.3 </td><td bgcolor=white> 37.5 </td><td bgcolor=white> 51.1 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3</th><td bgcolor=white> 608 </td><td bgcolor=white> 37.6 </td><td bgcolor=white> 59.4 </td><td bgcolor=white> 39.9 </td><td bgcolor=white> 20.4 </td><td bgcolor=white> 39.9 </td><td bgcolor=white> 48.2 </td></tr>
</table></tbody>

# Model

大家可以从下面的百度网盘链接来下载已训练好的模型：

链接: https://pan.baidu.com/s/1NmdqPwAmirknO5J__lg5Yw 

提起码: hlt6 

进入网盘中后，请点开网盘中的```weights/```文件夹，可以找到我们已训练好的模型权重。
