# A Discriminative Model with Multiple Temporal Scales for Action Prediction
(前情概要:作为阅读这个领域第一篇论文,虽然文章内容偏向于机器学习(SVM),和深度学习网络有一定区别,但文章设计思想值的借鉴,全局和局部信息分开处理,然后融合,这为后面模型有一定启发.在此之前阅读过一篇"TAM: TEMPORAL ADAPTIVE MODULE FOR VIDEO RECOGNITION",也是运用局部信息和全局信息的融合实现,可见这种思想对后面有一定启发.在查阅资料发现,本篇论文很少有人作出博客讲解,希望能分享自己鄙见.)

## 相关信息

论文名称: A Discriminative Model with Multiple Temporal Scales for Action Prediction
<br/>
论文作者: Yu Kong, Dmitry Kit, and Yun Fu1
<br/>
下载地址: https://projet.liris.cnrs.fr/imagine/pub/proceedings/ECCV-2014/papers/8693/86930596.pdf
<br/>

## 论文概述

所谓的动作预测,是在动作即将发生之前能够判断动作类型,而越早能预测出来对视频检测,灾难预警很极大意义.为了实现动作预测和提高视频开始部分的区别动作的能力,作者提出多重时序尺度的支持向量机(MTSSVM).论文围绕这样一个模型设计展开,简单来说模型既能读取"全局"信息,也能读取部分信息,模型在某些方面加了一些限制提高效率等,最终作者取得满意的效果.

## 论文要点

### 论文的作者想要完成什么，或者已经完成了什么？

论文目的在3.4节说明,如下图

![图 1](http://striveyadong.com/wp-content/uploads/2021/10/67f855b1ae594ebe3ef847a90fadae37c3230e0943877b35bf70f942f7f7149c-1.png)  

![图 2](http://striveyadong.com/wp-content/uploads/2021/10/414cfde8cfd7dd8017061e5a4e57b17766bc0149c7a9b0f343d0cc64139ffd85-1.png)  


通俗来说作者贡献有:

1. 设计一种多重时序尺度,即不是从单一角度来看待输入的数据(视频),而是结合了局部片段视频和全局获得的视频,对两方面进行评估获得不错的效果
2. 针对于时间的评价方法.在我的理解来看,如果取视频的1/2来预测和3/4来预测,那么3/4分数势必要比1/2的高(有可能后面的片段拉低了评分),但3/4视频评分机制是在基于1/2基础上作加,所以一定会大于.作者在这里面说明这会随着时间获得先验知识.同时之前的模型不同在于选自于动作事件发生的开始而文中针对于一个不完整的视频.
3. 片段级别的限制,这个读过论文可能会理解透彻一些,简单来说作者设计一些优化过程的限制条件能限制类别标签,动作评估,增强动作区分类别能力等,有助于最终效果.
4. 作者设计一个经验风险函数来优化模型,提升识别效果,即损失函数设计

### 如果一篇论文介绍了一种新方法/技术/方法，那么该新方法的关键要素是什么?

这里对上面方法做一个详细的解读
1. 作者对数据的处理,作者将一个视频中提取T帧,然后用$x_(1,k)$ 作为observation level,当k=T时代表输入为整个视频所提取出来的帧,$x_(t)$代表第t个帧.注意几个概念 progress level,observation ratio.前者表示k的大小,越大表示模型可观察部分越多.observation ratio表示k/t.
2. 第一点解决了数据部分问题,但输入仍然是图片,在当时卷积神经网络并没有大热,作者提出一个函数来表示部分帧和单个帧,论文中说明 $g(x_(1,k),1;k)$ which is the histogram of visual words contained in the entire partial video, starting from the first segment to the k-th segment,此外$g(x_(1,k),l)$  which is a histogram of visual words whose temporal locations are within the l-th segment.
3. 解决上述表示问题,作者指出了预测公式,符合大部分学习算法公式即:

   ![图 1](http://striveyadong.com/wp-content/uploads/2021/10/08e4d6953678f58331c0d1e6df1752a3fccd53480fca6f2d1716f283bd193cd1.png)

   公式符合常规理解,在学习出w参数情况下找出概率最大化的概率,和准确的y作比较.其中重点在于$\phi(x_(1,k),y)$如何表示.

   作者设计为如下

   ![图 3](http://striveyadong.com/wp-content/uploads/2021/10/2cda172c36e54e52b79792f7f91facabf7379ce376a3e97e52ad6733bbcb397c.png)  

   公式主要分为两部分,前一部分整体以k作为因变量即可推断出是对整体信息做出的计算.第二部分是一个求和过程,查看求和因子不难发现是对帧的遍历,所以推断出这部分是部分帧预测信息的总和.

4. 有了上述一个总的公式,常理逻辑便是对公式每一部分进行分开论述.首先是第一部分,论文中叫做Global Progress Model,对其功能描述有: indicates how likely the action class of an unfinished action video x(1,k) (at progress level k) is y.公式描述为:

   ![图 4](http://striveyadong.com/wp-content/uploads/2021/10/644685c4750024e20642d6dd2eca051475341a70d4a3ba0ff03a383de8925fcd.png)  

   式子中难以理解只有$\alpha_k^T$这一个部分,他的功能是 used to score the unfinished video x(1,k).在progress level为k的情况下,直方图产生维度为D,$a_k$表示在这个等价下对应分数矩阵,size为D ×|Y|.对于一个未完成视频那么他对应$\alpha$应该有k个,对于大小应该为D × K × |Y|.

   这一部分作用,论文也有阐述:Essentially, the GPM captures the action appearance changes as the progress level increases, and characterizes the entire action evolution over time.

5. 第二部分论文叫做Local Progress Model (LPM).主要功能:indicates how likely the action classes of all the temporal segments x(l) (l = 1, ··· , k) in an unfinished video x(1,k) are all y.即每个片段可能是y的可能性,公式表示为:

   ![图 5](http://striveyadong.com/wp-content/uploads/2021/10/d08a94fbb7cec293a3e405ffb9311954a80748ef8fe48c705af00b2adb824896.png)  

   在上面理解基础下理解$\beta$,a vector of size D×K×|Y| encoding the weights for the configurations between segments and action labels, with their corresponding segment evidence.

   这样一个函数从片段级别上说明了问题.

6. 文章学习策略基于the structured SVM,在上述总公式中有一个$w$待解决.作者列出求$w$是一个凸优化过程,公式如下:

   ![图 6](http://striveyadong.com/wp-content/uploads/2021/10/f663a9d6dd84f0449f576e5e8d9c1f90542c67c9c656962598a8585113ea4999.png)  

   这个公式确实没有详细看,这个公式形式符合标准的凸优化过程,重点关注于三个限制条件:
   > The slack variables ξ1i and the Constraint (6) are usually used in SVM constraints on the class labels.
   > 
   > Constraint (7) considers temporal action evolution over time
   > 
   > The slack variables ξ3i and the Constraint (8) are used to maximize the discriminability of segments x(k).

   三个公式是对模型优化过程,缺一不可.从下面实验看对准确率提升有显著性作用.

7. 最后作者介绍了经验风险,即Empirical Risk Minimization:

   ![图 7](http://striveyadong.com/wp-content/uploads/2021/10/3798d8d25f737a4e35c41a29a499b0e1ef1cf9dd5abdf4592e34385d6a7deb31.png)

   经验风险了解不是很多:sweat_smile:



### 论文中哪些内容对你有用?

由于方法的局限性,论文对于后面研究深度学习可能帮助不是很大,但我觉得研读后面发表论文来看,论文所做出的的贡献值的参考的主要有: 
1. 一种分帧的思想,后面大多数视频任务也都是采用均匀采样.
2. 整体和部分分开求取计算,最终融合.这样避免部分片段影响最终结果等情况.

### 你还想关注哪些参考文献?

由于论文发表比较早,暂时没有想要看的参考文献.:sweat_smile:

## 相关code

暂未发现 :sweat_smile:





