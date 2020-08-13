## 机甲大师赛的目标装甲识别——哨兵视角

### 算法原理：
**根据观察，装甲两侧的灯条特征较为明显，可以进行特征提取和匹配。目前使用到的特征：**
- 灯条高度：两侧灯条的矩形面积在正对情况下可以明显区别与其他位置的灯条，但是侧面观察发现不管是不是同一装甲模块的灯条，靠近摄像头的部分灯条面积比远离摄像头的灯条面积大
    - 修正：灯条矩形宽的像素误差很大，导致矩形的面积误差放大。这样判断的结果存在很大问题，改为使用矩形的高作为依据具有更好的鲁棒性。
- 角度差距：如果是同一个装甲的灯条，那么不管如何移动2个灯条之间角度理论上都是相等的且受像素误差影响小，因此角度可以作为判断的重要条件
- 点线距离：同一装甲的两个灯条是平行的且长度相等。如果在3维空间中连接两个灯条的中垂线是同一条直线，但是映射到二维平面上会因为投影角度产生一定偏差，所以这种方法本身存在一定误差，对像素误差产生的角度误差也比较敏感。
- 连续性验证：两个灯条之间的距离在世界坐标中是固定，在图像坐标中会根据运动动态改变，因为不会突变可以用来辅助验证，增强确定目标的鲁棒性
- 预测位置：根据kalman滤波估测下一帧的位置，如果下一个位置和预测结果相近则认为之前匹配的是正确的，增强鲁棒性

- **可见，有很多的特征都可以用于装甲识别。对此可以分别对各项指标进行加权，最后取出权值最高的一对点作为最优目标**
- 在最后我还训练了一个神经网络模型再次过滤
### 流程图
![img](https://github.com/polomonk/RM_armor_detection/blob/master/images/flow.png)
<!--
```flow
st=>start: 开始
io=>inputoutput: 读入图像
op1=>operation: 图像预处理并找到灯条轮廓
op2=>operation: 对灯条轮廓分别加权并计算权重
op3=>operation: 删除保存的轮廓数据
cond=>condition: 权重是否大于阈值
sub1=>subroutine: 保存最优轮廓
e=>end: 结束

st->io->op1->op2->cond
cond(no)->op3->e
cond(yes)->sub1(left)->op2
```
-->
### 识别结果
![img](https://github.com/polomonk/RM_armor_detection/blob/master/images/dst.png)
- 粉色的点为预测打击的目标
