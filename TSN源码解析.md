# TSN论文阅读和源码解析

## 背景介绍

### 论文相关

论文名称:Temporal Segment Networks for Action Recognition in Videos

论文地址:<https://arxiv.org/abs/1705.02953>

论文作者:Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool

项目地址1: <https://github.com/yjxiong/temporal-segment-networks>

项目地址2: <https://github.com/yjxiong/tsn-pytorch(pytorch>版本)

### 项目结构(pytorch版本)

TSN项目结构如下所示

```shell
TSN
├─ .gitignore
├─ .gitmodules
├─ dataset.py      --数据集加载文件
├─ LICENSE
├─ main.py         --主文件,主函数
├─ models.py       --模型文件
├─ ops
│  ├─ basic_ops.py
│  ├─ utils.py
│  └─ __init__.py
├─ opts.py         --可修改选项
├─ README.md
├─ test_models.py  --测试模型
├─ tf_model_zoo
└─ transforms.py

```

### 环境配置

## 论文要点

## 项目解析

### model.py(模型文件)

#### 类参数
 

#### get_optim_policies(self)函数

函数主要是获取模型每一层参数并以字典形式返回参数.返回部分代码如下,第一层和其余的区分对待
逻辑:是不是卷积层(是的话是不是第一层)->是不是线性层->是不是BN层

```python
return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]
```

### main.py(主文件)
