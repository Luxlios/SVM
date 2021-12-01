# SVM
手写SVM对Iris和Sonar数据集分类

数据集：  
[Sonar](http://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))  
[Iris](http://archive.ics.uci.edu/ml/datasets/Iris)

#### Content
- [理论推导](#SVM简单推导)
- [实验结果](#实验结果)

##### SVM简单推导

SVM算法具有严格的理论推导，有较强的数学意义，下面简单的展示一下SVM求解算法的推导过程，具体推导过程可参考《UNDERSTANDING MACHINE LEARNING FROM THEORY TO ALGORITHMS》。
<div align="center">
  <img src="https://github.com/Luxlios/Figure/blob/main/SVM/derivation1.png" height="400">
</div>

上图是硬聚类和软聚类的目标函数和约束条件，硬聚类就是经典的SVM算法形式，建立在数据线性可分的情况下。而现实生活中，数据常常不可分，为了解决这个问题，允许部分样本出错，因此引入了松弛变量和惩罚因子C，得到了SVM软聚类的算法形式。  

关于惩罚因子C，值越大，表示后者的权重越大，即越不希望出现错误分类的情况；C越小，表示前者的权重越大，允许样本出错，争取omiga的二范数越小。  
remark：当C取无穷时，表示后者占据主要地位，不允许出现错误分类，软聚类转化为硬聚类。

对于SVM软聚类算法形式，我们可以用lagrange乘数法进行化简。两个约束条件转化为如下形式。
<div align="center">
  <img src="https://github.com/Luxlios/Figure/blob/main/SVM/derivation2.png" height="150">
</div>

目标函数转化为如下形式。
<div align="center">
  <img src="https://github.com/Luxlios/Figure/blob/main/SVM/derivation3.png" height="50">
</div>

将两个约束条件代入得：
<div align="center">
  <img src="https://github.com/Luxlios/Figure/blob/main/SVM/derivation4.png" height="60">
</div>

满足KKT条件的话，下面这两个式子可以相互转化（可以证明其满足，这里不赘述）。
<div align="center">
  <img src="https://github.com/Luxlios/Figure/blob/main/SVM/derivation5.png" height="70">
</div>

转化为对偶问题后，分别对三个变量求导并令等式为0。
<div align="center">
  <img src="https://github.com/Luxlios/Figure/blob/main/SVM/derivation6.png" height="200">
</div>

代入可求得里面那层的最小值，得到如下式子。
<div align="center">
  <img src="https://github.com/Luxlios/Figure/blob/main/SVM/derivation7.png" height="80">
</div>

用核方法只需把内积以如下式子变化，证明过程这里不赘述。
<div align="center">
  <img src="https://github.com/Luxlios/Figure/blob/main/SVM/derivation8.png" height="110">
</div>

##### 实验结果
以线性核函数为例。
<div align="center">
  <img src="https://github.com/Luxlios/Figure/blob/main/SVM/result.png" height="200">
</div>


