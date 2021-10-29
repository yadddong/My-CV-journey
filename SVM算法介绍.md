# SVM算法介绍

(转载自博客[看了这篇文章你还不懂SVM你就来打我](https://tangshusen.me/2018/10/27/SVM/),请大家支持正版博客,本博客仅做收藏使用)

## 概要

### 简介

SVM是什么? 
>支持向量机（英语：support vector machine，常简称为SVM，又名支持向量网络）是在分类与回归分析中分析数据的监督式学习模型与相关的学习算法。给定一组训练实例，每个训练实例被标记为属于两个类别中的一个或另一个，SVM训练算法创建一个将新的实例分配给两个类别之一的模型，使其成为非概率二元线性分类器。SVM模型是将实例表示为空间中的点，这样映射就使得单独类别的实例被尽可能宽的明显的间隔分开。然后，将新的实例映射到同一空间，并基于它们落在间隔的哪一侧来预测所属类别。

简单点讲，SVM就是一种二类分类模型，他的基本模型是的定义在特征空间上的间隔最大的线性分类器，SVM的学习策略就是间隔最大化。

### 直观理解

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/1.1.png" width="40%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">图1-1</div>
</center>
<div STYLE="page-break-after: always;"></div>

图中有分别属于两类的一些二维数据点和三条直线。如果三条直线分别代表三个分类器的话，请问哪一个分类器比较好？

我们凭直观感受应该觉得答案是H3。首先H1不能把类别分开，这个分类器肯定是不行的；H2可以，但分割线与最近的数据点只有很小的间隔，如果测试数据有一些噪声的话可能就会被H2错误分类(即对噪声敏感、泛化能力弱)。H3以较大间隔将它们分开，这样就能容忍测试数据的一些噪声而正确分类，是一个泛化能力不错的分类器。

对于支持向量机来说，数据点若是p维向量，我们用p−1维的超平面来分开这些点。但是可能有许多超平面可以把数据分类。最佳超平面的一个合理选择就是以最大间隔把两个类分开的超平面。因此，SVM选择能够使离超平面最近的数据点的到超平面距离最大的超平面。

以上介绍的SVM只能解决线性可分的问题，为了解决更加复杂的问题，支持向量机学习方法有一些由简至繁的模型:

- 线性可分SVM
  > 当训练数据线性可分时，通过硬间隔(hard margin，什么是硬、软间隔下面会讲)最大化可以学习得到一个线性分类器，即硬间隔SVM，如上图的的H3。
- 线性SVM
  > 当训练数据不能线性可分但是可以近似线性可分时，通过软间隔(soft margin)最大化也可以学习到一个线性分类器，即软间隔SVM。
- 非线性SVM
  > 当训练数据线性不可分时，通过使用核技巧(kernel trick)和软间隔最大化，可以学习到一个非线性SVM。

## 线性可分SVM——硬间隔

考虑如下形式的线性可分的训练数据集:
$${(X_1,y_1),(X_2,y_2),…,(X_n,y_n)}$$

其中$Xi$是一个含有$d$个元素的列向量, 即$X_i\in \mathbf{R}^d
$; $y_i$是标量, $y\in+1,−1$, $y_i=+1$时表示$X_i$属于正类别, $y_i=−1$时表示$X_i$属于负类别。

> 注: 本文中, $X、X_i、W$等都是(列)向量，有的文章一般用 $x_i$ 表示一个向量而用 $X$ 表示所有 $x_i$ 组成的一个矩阵，注意区分。

回忆一下感知机的目标: 找到一个超平面使其能正确地将每个样本正确分类。感知机使用误分类最小的方法求得超平面，不过此时解有无穷多个(例如图1.1的H2和H3以及它俩的任意线性组合)。而线性可分支持向量机利用间隔最大化求最优分离超平面,这时解是唯一的。

### 超平面和间隔

一个超平面由法向量$W$和截距$b$决定,其方程为$X^TW+b=0$, 可以规定法向量指向的一侧为正类,另一侧为负类。下图画出了三个平行的超平面，法方向取左上方向。

> 注意: 如果$X$和$W$都是列向量,即$X^TW$会得到$X$和$W$的点积(dot product, 是一个标量),等价于$X⋅W$和$W⋅X$。

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/2.1.png" width="40%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">图2-1</div>
</center>
<div STYLE="page-break-after: always;"></div>

为了找到最大间隔超平面，我们可以先选择分离两类数据的两个平行超平面，使得它们之间的距离尽可能大。在这两个超平面范围内的区域称为“间隔(margin)”，最大间隔超平面是位于它们正中间的超平面。这个过程如上图所示。

### 间隔最大化

将高数里面求两条平行直线的距离公式推广到高维可求得图2.1中margin的$ρ$:

$$margin = \rho = \frac 2 {||W||} \tag{2.2.1}$$

我们的目标是使$ρ$最大, 等价于使$ρ^2$最大:

$$\underset{W,b}{max} \rho \iff \underset{W,b}{max} \rho^2 \iff \underset{W,b}{min}\frac 1 2 ||W||^2 \tag{2.2.2}$$

上式的$\frac1 2$是为了后续求导后刚好能消去，没有其他特殊意义。

同时也不要忘了有一些约束条件:

$$X_i^TW+b \ge +1, y_i=+1 \\\\ X_i^TW+b \le -1, y_i=-1 \tag{2.2.3}$$

总结一下，间隔最大化问题的数学表达就是

$$\underset{W,b}{min}J(W) = \underset{W,b}{min}\frac 1 2 ||W||^2 \\\\ s.t.\quad y_i(X_i^TW+b) \ge 1, i=1,2,…n. \tag{2.2.4}$$

通过求解上式即可得到最优超平面 $\hat{W}$ 和 $\hat{b}$ 。具体如何求解下文。

### 支持向量

在线性可分的情况下，训练数据集的样本点中与分离超平面距离最近的数据点称为支持向量(support vector)，支持向量是使$(2.2.4)$中的约束条件取等的点，即满足
$$y_i(X_i^TW+b) = 1 \tag{2.3.1}$$

的点。也即所有在直线$X^TW+b=1$或直线$X^TW+b=−1$的点。如下图所示:

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/2.2.png" width="40%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">图2-2</div>
</center>
<div STYLE="page-break-after: always;"></div>

**在决定最佳超平面时只有支持向量起作用，而其他数据点并不起作用。**如果移动非支持向量，甚至删除非支持向量都不会对最优超平面产生任何影响。也即支持向量对模型起着决定性的作用，这也是“支持向量机”名称的由来。

### 对偶问题

如何求解式 $(2.2.4)$ 呢？

我们称式 $(2.2.4)$ 所述问题为原始问题(primal problem), 可以应用拉格朗日乘子法构造拉格朗日函数(Lagrange function)再通过求解其对偶问题(dual problem)得到原始问题的最优解。转换为对偶问题来求解的原因是:

> - 对偶问题更易求解，由下文知对偶问题只需优化一个变量$α$且约束条件更简单；
> 
> - 能更加自然地引入核函数，进而推广到非线性问题。

首先构建拉格朗日函数。为此需要引进拉格朗日乘子(Lagrange multiplier) $α_i≥0,i=1,2,…n。$则拉格朗日函数为:
$$L(W,b,\alpha)=\frac 1 2 ||W||^2 - \sum_{i=1}^n \alpha_i [y_i(X_i^TW+b)-1] \tag{2.4.1}$$
因此，给定一个$W$和$b$, 若不满足式$(2.2.4)$的约束条件，那么有
$$\underset{\alpha}{max} L(W,b,\alpha) = +\infty \tag{2.4.2}$$
否则，若满足式$(2.2.4)$的约束条件，有
$$\underset{\alpha}{max} L(W,b,\alpha) = J(W) = \frac 1 2 ||W||^2 \tag{2.4.3}$$
结合式$(2.4.2)$和$(2.4.3)$知，优化问题
$$\underset{W, b}{min} \underset{\alpha}{max} L(W,b,\alpha)\tag{2.4.4}$$
与式$(2.2.4)$所述问题是完全等价的。

根据拉格朗日对偶性，式$(2.4.4)$所述问题即原始问题的对偶问题是:
$$\underset{\alpha}{max} \underset{W, b}{min} L(W,b,\alpha) \tag{2.4.5}$$
为了求得对偶问题的解，需要先求得$L(W,b,α)$对$W$和$b$的极小再求对α的极大。

(1) 求$\underset{W, b}{min} L(W,b,\alpha)$:
$$\nabla_W L(W,b,\alpha) = W - \sum_{i=1}^n \alpha_i y_i X_i = 0 \implies W= \sum_{i=1}^n \alpha_i y_i X_i\tag{2.4.6}$$
$$\nabla_b L(W,b,\alpha) = - \sum_{i=1}^n \alpha_i y_i = 0 \implies \sum_{i=1}^n \alpha_i y_i = 0 \tag{2.4.7}$$
将上面两式代入$L(W,b,\alpha)$:
<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/fomula_1.png" width="60%" height="60%" />
</div>
<!-- <center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;"></div>
</center> -->
<div STYLE="page-break-after: always;"></div>
所以，

$$\underset{W, b}{min} L(W,b,\alpha) = -\frac 1 2 \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j X_i^T X_j \ + \ \sum_{i=1}^n \alpha_i \tag{2.4.8}$$

(2) 求$\underset{W, b}{min} L(W,b,\alpha)$对α的极大:
等价于式$(2.4.8)$对$α$求极大，也等价于式$(2.4.8)$取负数后对$α$求极小，即
$$\underset{\alpha}{min}  \quad  \frac 1 2 \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j X_i^T X_j \ - \ \sum_{i=1}^n \alpha_i  \tag{2.4.9}$$
同时满足约束条件:
$$\sum_{i=1}^n \alpha_i y_i = 0 \\\\ \alpha_i \ge 0, i=1,2,…,n. \tag{2.4.10}$$

至此，我们得到了原始最优化问题$(2.2.4)$和对偶最优化问题$(2.4.9)$、$(2.4.10)$。
由slater条件知，因为原始优化问题的目标函数和不等式约束条件都是凸函数，并且该不等式约束是严格可行的(因为数据是线性可分的), 所以存在$\hat{W}$$\hat{b}$$\hat{a}$,使得$\hat{W}$$\hat{b}$是原始问题的解，$\hat{a}$是对偶问题的解。这意味着求解原始最优化问题(2.2.4)可以转换为求解对偶最优化问题(2.4.9)、(2.4.10)。
> slater 条件:
> 原始问题一般性表达为
> $$\underset{x}{min} \quad f(x) \\\\ s.t. \ c_i(x) \le 0, i=1,2,…k \\\\ \quad \quad h_j(x) = 0, j=1,2,…,l$$
> 则其拉格朗日函数为
> $$L(x,\alpha,\beta)=f(x) + \sum_{i=1}^k \alpha_i c_i(x) + \sum_{j=1}^l \beta_j h_j(x), \quad \alpha_i \ge 0$$
> 假设原始问题目标函数 $f(x)$ 和不等式约束条件 $c_i(x)$都是凸函数，原始问题等式约束$hj(x)$都是仿射函数，且不等式约束 $c_i(x)$是严格可行的，即存在 x ，对所有 i 都有 $c_i(x)<0$ ，则存在$\hat{x}$, $\hat{\alpha}$, $\hat{\beta}$，使$\hat{x}$是原始问题的解，$\hat{\alpha}$, $\hat{\beta}$是对偶问题的解。

那么如何求解优化问题(2.4.9)、(2.4.10)的最优解 $\hat{\alpha}$ 呢？
不难发现这是一个二次规划问题，有现成的通用的算法来求解。
> 事实上通用的求解二次规划问题的算法的复杂度正比于训练数据样本数，所以在实际应用中需要寻求更加高效的算法，例如序列最小优化(Sequential Minimal Optimiation, SMO)算法。

假设我们现在求得了(2.4.9)、(2.4.10)的最优解 $\hat{\alpha}$，则根据式(2.4.6)可求得最优$\hat{W}$：
$$\hat{W}= \sum_{i=1}^n \hat{\alpha}_i y_i X_i \tag{2.4.11}$$
因为至少存在一个$\hat{\alpha}_i$ (若不存在，即 $\hat{\alpha}$ 全为0，则 $\hat{W}=0$, 即 $margin = \frac 2 {||\hat{W}||}= \infty$,显然不行), 再根据KKT条件，即
$$\begin{cases} 乘子非负: \alpha_i \ge 0 (i=1,2,…n.下同) \\\\ 约束条件:  y_i(X_i^TW+b) - 1\ge 0 \\\\ 互补条件:  \alpha_i (y_i(X_i^TW+b) - 1)=0 \end{cases}$$

