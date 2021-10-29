# 论文背景

论文地址: https://arxiv.org/abs/1312.6114

论文名称: Auto-Encoding Variational Bayes

论文作者: Diederik P Kingma, Max Welling

解决问题:如何构造编码器和解码器，使得图片能够编码成易于表示的形态，并且这一形态能够尽可能无损地解码回原真实图像。

# 先验知识

## 生成模型

在概率统计理论中， 生成模型是指能够随机生成观测数据的模型，尤其是在给定某些隐含参数的条件下。它给观测值和标注数据序列指定一个联合概率分布。在机器学习中，生成模型可以用来直接对数据建模（例如根据某个变量的概率密度函数进行数据采样），也可以用来建立变量间的条件概率分布。条件概率分布可以由生成模型根据贝叶斯定理形成。[来源于百度百科](https://baike.baidu.com/item/%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B/6563656)

## 贝叶斯公式
首先抛出贝叶斯推断（Bayesian inference）:
$$ p(\theta|x)=\frac{p(x|\theta)p(\theta)}{p(x)}$$
其中$\theta$代表一种假设,$x$代表一种观察结果

式子中$p(\theta)$为先验概率，是在还没有观测$x$的情况下,$\theta$自身的概率。

称$p(\theta|x)$为后验概率，表示在观察到了$x$的情况下，$\theta$的条件概率。

称$L(x|\theta)=C\dot{p(x|\theta)}$为似然函数（不要叫似然概率），其中C为常数，因为似然函数的绝对数值没有意义。

称$p(x)$为证据因子(model evidence)，有时也会称为边缘似然。

## PCA
<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/pca.jpg" width="100%" height="50%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">PCA基本思想</div>
</center>
<div STYLE="page-break-after: always;"></div>

如图，X本身是一个矩阵，通过一个变换W变成了一个低维矩阵c，因为这一过程是线性的，所以再通过一个$W^T$变换就能还原出一个，现在我们要找到一种变换W，使得矩阵X与能够尽可能地一致，这就是PCA做的事情。 如图，X本身是一个矩阵，通过一个变换W变成了一个低维矩阵c，因为这一过程是线性的，所以再通过一个变换就能还原出一个$\hat{x}$，现在我们要找到一种变换W，使得矩阵X与$\hat{x}$能够尽可能地一致，这就是PCA做的事情。

现在我们需要对这一雏形进行改进。首先一个最明显能改进的地方是用神经网络代替W变换和$W^T$变换，就得到了如下Deep Auto-Encoder模型：

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/deep_decoder.jpg" width="100%" height="60%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">Deep Auto-Encoder</div>
</center>
<div STYLE="page-break-after: always;"></div>

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/deep_codercnn.jpg" width="100%" height="60%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">Deep Auto-Encoder例子</div>
</center>
<div STYLE="page-break-after: always;"></div>

# 设计思想
## 论文思路
把一堆真实样本通过编码器网络变换成一个理想的数据分布，然后这个数据分布再传递给一个解码器网络，得到一堆生成样本，生成样本与真实样本足够接近的话，就训练出了一个自编码器模型。那VAE(变分自编码器)就是在自编码器模型上做进一步变分处理，使得编码器的输出结果能对应到目标分布的均值和方差，如下图所示，
<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/基本思想.jpg" width="100%" height="60%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">基本思想</div>
</center>
<div STYLE="page-break-after: always;"></div>

## 实际问题

上述我们构造出了一个重构图像比较清晰的自编码模型，但是这并没有达到我们真正想要构造的生成模型的标准，因为，对于一个生成模型而言，解码器部分应该是单独能够提取出来的，并且对于在规定维度下任意采样的一个编码，都应该能通过解码器产生一张清晰且真实的图片。

我们先来分析一下现有模型无法达到这一标准的原因。

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/PRO.jpg" width="100%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">问题</div>
</center>
<div STYLE="page-break-after: always;"></div>

如上图所示，假设有两张训练图片，一张是全月图，一张是半月图，经过训练我们的自编码器模型已经能无损地还原这两张图片。接下来，我们在code空间上，两张图片的编码点中间处取一点，然后将这一点交给解码器，我们希望新的生成图片是一张清晰的图片（类似3/4全月的样子）。但是，实际的结果是，生成图片是模糊且无法辨认的乱码图。一个比较合理的解释是，因为编码和解码的过程使用了深度神经网络，这是一个非线性的变换过程，所以在code空间上点与点之间的迁移是非常没有规律的。

如何解决这个问题呢？我们可以引入噪声，使得图片的编码区域得到扩大，从而掩盖掉失真的空白编码点。
<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/s1.jpg" width="100%" height="60%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">方案</div>
</center>
<div STYLE="page-break-after: always;"></div>

如上图所示，现在在给两张图片编码的时候加上一点噪音，使得每张图片的编码点出现在绿色箭头所示范围内，于是在训练模型的时候，绿色箭头范围内的点都有可能被采样到，这样解码器在训练时会把绿色范围内的点都尽可能还原成和原图相似的图片。然后我们可以关注之前那个失真点，现在它处于全月图和半月图编码的交界上，于是解码器希望它既要尽量相似于全月图，又要尽量相似于半月图，于是它的还原结果就是两种图的折中（3/4全月图）。

由此我们发现，给编码器增添一些噪音，可以有效覆盖失真区域。不过这还并不充分，因为在上图的距离训练区域很远的黄色点处，它依然不会被覆盖到，仍是个失真点。为了解决这个问题，我们可以试图把噪音无限拉长，使得对于每一个样本，它的编码会覆盖整个编码空间，不过我们得保证，在原编码附近编码的概率最高，离原编码点越远，编码概率越低。在这种情况下，图像的编码就由原先离散的编码点变成了一条连续的编码分布曲线，如下图所示。

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/s2.jpg" width="100%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">方案</div>
</center>
<div STYLE="page-break-after: always;"></div>

那么上述的这种将图像编码由离散变为连续的方法，就是变分自编码的核心思想。下面就会介绍VAE的模型架构，以及解释VAE是如何实现上述构思的。

## 详细设计

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/base.jpg" width="100%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">基本架构</div>
</center>
<div STYLE="page-break-after: always;"></div>

上面这张图就是VAE的模型架构，我们先粗略地领会一下这个模型的设计思想。

在auto-encoder中，编码器是直接产生一个编码的，但是在VAE中，为了给编码添加合适的噪音，编码器会输出两个编码，一个是原有编码($m_1,m_2,m_3$)，另外一个是控制噪音干扰程度的编码($\sigma_1,\sigma_2,\sigma_3$)，第二个编码其实很好理解，就是为随机噪音码($e_1,e_2,e_3$)分配权重，然后加上exp($\sigma_i$)的目的是为了保证这个分配的权重是个正值，最后将原编码与噪音编码相加，就得到了VAE在code层的输出结果($c_1,c_2,c_3$)。其它网络架构都与Deep Auto-encoder无异。

损失函数方面，除了必要的重构损失外，VAE还增添了一个损失函数（见上图Minimize2内容），这同样是必要的部分，因为如果不加的话，整个模型就会出现问题：为了保证生成图片的质量越高，编码器肯定希望噪音对自身生成图片的干扰越小，于是分配给噪音的权重越小，这样只需要将($\sigma_1,\sigma_2,\sigma_3$)赋为接近负无穷大的值就好了。所以，第二个损失函数就有限制编码器走这样极端路径的作用，这也从直观上就能看出来，exp($\sigma_i$)-(1+$\sigma_i$)在=0处取得最小值，于是($\sigma_1,\sigma_2,\sigma_3$)就会避免被赋值为负无穷大。

上述我们只是粗略地理解了VAE的构造机理，但是还有一些更深的原理需要挖掘，例如第二个损失函数为何选用这样的表达式，以及VAE是否真的能实现我们的预期设想，即“图片能够编码成易于表示的形态，并且这一形态能够尽可能无损地解码回原真实图像”，是否有相应的理论依据。

下面我们会从理论上深入地分析一下VAE的构造依据以及作用原理。

# VAE原理

## 理解层面
我们知道，对于生成模型而言，主流的理论模型可以分为隐马尔可夫模型HMM、朴素贝叶斯模型NB和高斯混合模型GMM，而VAE的理论基础就是高斯混合模型。

什么是高斯混合模型呢？就是说，任何一个数据的分布，都可以看作是若干高斯分布的叠加。

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/t.jpg" width="100%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">理论</div>
</center>
<div STYLE="page-break-after: always;"></div>

如图所示，如果P(X)代表一种分布的话，存在一种拆分方法能让它表示成图中若干浅蓝色曲线对应的高斯分布的叠加。有意思的是，这种拆分方法已经证明出，当拆分的数量达到512时，其叠加的分布相对于原始分布而言，误差是非常非常小的了。

于是我们可以利用这一理论模型去考虑如何给数据进行编码。一种最直接的思路是，直接用每一组高斯分布的参数作为一个编码值实现编码。

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/gs_p.jpg" width="100%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">离散</div>
</center>
<div STYLE="page-break-after: always;"></div>

如上图所示，m代表着编码维度上的编号，譬如实现一个512维的编码，m的取值范围就是1,2,3……512。m会服从于一个概率分布P(m)（多项式分布）。现在编码的对应关系是，每采样一个m，其对应到一个小的高斯分布$N(\mu^i,\sigma_i)$，P(X)就可以等价为所有的这些高斯分布的叠加，即：
$$P(x)=\sum_m P(m)P(x|m) $$

其中m~P(m),x|m~$N(\mu^i,\sigma_i)$

上述的这种编码方式是非常简单粗暴的，它对应的是我们之前提到的离散的、有大量失真区域的编码方式。于是我们需要对目前的编码方式进行改进，使得它成为连续有效的编码。

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/gs.jpg" width="100%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">连续</div>
</center>
<div STYLE="page-break-after: always;"></div>

现在我们的编码换成一个连续变量z，我们规定z服从正态分布$N(0,1)$（实际上并不一定要选用，其他的连续分布都是可行的）。每对于一个采样z，会有两个函数和，分别决定z对应到的高斯分布的均值和方差，然后在积分域上所有的高斯分布的累加就成为了原始分布P(X),即：
$$P(x)=\int_z P(z)P(x|z)dz$$
其中$z~N(0,1)$,$x|z~N(\mu(z),\sigma(z))$

接下来就可以求解这个式子。由于是P(z)已知的，P(x|z)未知，而$x|z~N(\mu(z),\sigma(z))$于是我们真正需要求解的，是$\mu(z)$和$\sigma(z)$两个函数的表达式。又因为P(x)通常非常复杂，导致和难以计算，我们需要引入两个神经网络来帮助我们求解。

第一个神经网络叫做Decoder，它求解的是$\mu(z)$和$\sigma(z)$两个函数，这等价于$P(x|z)$求解。

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/decoder.jpg" width="100%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">decoder</div>
</center>
<div STYLE="page-break-after: always;"></div>

第二个神经网络叫做Encoder，它求解的结果是$q(z|x)$，q可以代表任何分布。

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/encoder.jpg" width="100%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">encoder</div>
</center>
<div STYLE="page-break-after: always;"></div>

 值得注意的是，这儿引入第二个神经网路Encoder的目的是，辅助第一个Decoder求解$P(x|z)$


## 数学层面
推导见下图

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/p1.jpg" width="100%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">p1</div>
</center>
<div STYLE="page-break-after: always;"></div>

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/p2.jpg" width="100%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">p2</div>
</center>
<div STYLE="page-break-after: always;"></div>

# 代码实现

代码参照[github项目](https://github.com/wiseodd/generative-models),代码逻辑很简单

```python
import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


# =============================== Q(z|X) ======================================

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Whz_mu = xavier_init(size=[h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

Whz_var = xavier_init(size=[h_dim, Z_dim])
bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)


def Q(X):
    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
    z_mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
    z_var = h @ Whz_var + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var


def sample_z(mu, log_var):
    eps = Variable(torch.randn(mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)


def P(z):
    h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X


# =============================== TRAINING ====================================

params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
          Wzh, bzh, Whx, bhx]

solver = optim.Adam(params, lr=lr)

for it in range(100000):
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))

    # Forward
    z_mu, z_var = Q(X)
    z = sample_z(z_mu, z_var)
    X_sample = P(z)

    # Loss
    recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / mb_size
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    loss = recon_loss + kl_loss

    # Backward
    loss.backward()

    # Update
    solver.step()

    # Housekeeping
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))

        samples = P(z).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c += 1
        plt.close(fig)

```

最终在colab上运行结果:
<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/r1.jpg" width="100%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">r1</div>
</center>


<div STYLE="page-break-after: always;"></div>
<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/r2.png" width="100%" height="40%" />
</div>
<center>
<div style="color:orange; border-bottom: 1px solid #d9d9d9;
    align=center;
    display: inline-block;
    color: #999;
    padding: 2px;">r2</div>
</center>
<div STYLE="page-break-after: always;"></div>

最后感谢博客[变分自编码器（VAEs）](https://zhuanlan.zhihu.com/p/25401928)
[生成模型——变分自编码器](http://www.gwylab.com/note-vae.html)

博客绝大部分内容源于他们,对理解其论文起到很大帮助