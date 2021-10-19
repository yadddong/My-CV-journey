# Saliency Tubes---时空卷积神经网络可视化

## 项目相关
### 文章地址
论文地址: https://arxiv.org/pdf/1902.01078.pdf

论文作者: Alexandros Stergiou, Georgios Kapidis,Grigorios Kalliatakis, Christos Chrysoulas, Remco Veltkamp, Ronald Poppe

项目地址: https://github.com/alexandrosstergiou/Saliency-Tubes-Visual-Explanations-for-Spatio-Temporal-Convolutions

### 项目效果展示
<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/bowling.gif" width="50%" height="50%" />
</div>
<div STYLE="page-break-after: always;"></div>

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/cliff_diving.gif" width="50%" height="50%" />
</div>
<div STYLE="page-break-after: always;"></div>

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/opening_door.gif" width="50%" height="50%" />
</div>
<div STYLE="page-break-after: always;"></div>

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/opening_drawer.gif" width="50%" height="50%" />
</div>
<div STYLE="page-break-after: always;"></div>

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/rafting.gif" width="50%" height="50%" />
</div>
<div STYLE="page-break-after: always;"></div>

<div align=center>
<img src="http://striveyadong.com/wp-content/uploads/2021/10/washing.gif" width="50%" height="50%" />
</div>
<div STYLE="page-break-after: always;"></div>

## 预先准备

在进行可视化之前需要安装的库有:OpenCV,moviepy,FFmpeg,scipy,执行以下指令:
```
pip install opencv-python
pip install scipy
pip install moviepy
```

其中Saliency Tubes地址为[Saliency Tubes](https://github.com/alexandrosstergiou/Saliency-Tubes-Visual-Explanations-for-Spatio-Temporal-Convolutions),通过指令下载到本地:
```
git clone https://github.com/alexandrosstergiou/Saliency-Tubes-Visual-Explanations-for-Spatio-Temporal-Convolutions.git
```
(默认认为电脑已经配置pytorch或者TensorFlow等基本深度学习环境)

下载后文件结构:
```
Saliency-Tubes-orginal --项目名称
├─ .gitignore
├─ examples   --展示案例
│  ├─ bowling.gif
│  ├─ cliff_diving.gif
│  ├─ opening_door.gif
│  ├─ opening_drawer.gif
│  ├─ rafting.gif
│  └─ washing.gif
├─ frames.py  --分帧文件
├─ heat_tubes_Keras.py          --keras框架下热图生成
├─ heat_tubes_mfnet_pytorch.py  --pytorch下的热图加载文件
├─ LICENSE
├─ mfnet_3d.py   --示例的模型文件
└─ README.md

```

## 文件详解

### frames .py(分帧文件)

- frames_extractor(file,frame_num,step)函数:
  参数 1. file:文件名
       2. frame:最大分帧数
       3. step:分帧间隔(每隔step整数倍记录一次)
   核心代码:
   ```python
    img = frame[y/2-112:y/2+112,x/2-112:x/2+112]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.true_divide(img,255.0)
   ```
   np.true_devide解析:方法主要是前一个数组除以后面的数组,这里显然使用了广播的机制,相当于做了一个归一化

   代码第一句主要作用是把图片规格限定在224*224

### heat_tubes_mfnet_pytorch.py(核心文件,生成热图)

- 运行时的几个参数:
  ```python
  parser.add_argument("num_classes", type=int)
  parser.add_argument("model_weights", type=str)
  parser.add_argument("frame_dir", type=str)
  parser.add_argument("label", type=int)
  parser.add_argument("--base_output_dir", type=strdefault=r"visualisations")
  ```
  1. num_classes:模型文件需要的类别数量
  2. model_weights:模型训练好参数地址
  3. frame_dir:帧图片地址
  4. label:
  5. base_output_dir:输出地址
- 选帧条件:代码事先加载帧文件夹下所有文件,通过np.linespace均匀取16个数据,形成indice数组,最终以这个数组作为标准取帧图片.
  ```python
  frame_names = os.listdir(args.frame_dir)frame_indices = list(np.linspace(0, len(frame_names)-1, num=16, dtype=np.int))selected_frames = [frame_names[i] for i in frame_indices]
  ```
  


- load_images(args.frame_dir, selected_frames):加载图片函数
  参数:frame_dir:帧地址文件夹
       select_frames:选取的帧图片文件名
  
  该函数主要是图片规格进行规范化,首先通过迭代形式把指定目录下文件进行修剪,最终保存格式为256*256,随后对总的图片列表进行归一化操作,最终返回结果对维度进行扩充.返回的结果一个是原图片(没有cv2.COLOR_BGR2RGB之前),一个是修剪之后图片.

  图片归一化.
  ```python
    torch_imgs = torch.from_numpy(images.transpose(3,0,1,2))
    torch_imgs = torch_imgs.float() / 255.0
    mean_3d = [124 / 255, 117 / 255, 104 / 255]
    std_3d = [0.229, 0.224, 0.225]
  ```