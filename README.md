# 宠物头部检测实验

本项目基于YOLOv5s实现了宠物（猫狗）头部区域的目标检测。使用Oxford-IIIT宠物数据集进行训练和评估。

## 数据集

数据来源：[Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)

- 数据集包含超过7000张猫狗图像，其中约3700张图像有头部区域标注
- 标注采用PASCAL VOC格式
- 数据已预处理为YOLO格式，分为训练集和验证集（测试集）

## 文件结构

- `config.yaml` - 配置文件，包含训练和评估参数
- `train.py` - 训练脚本
- `evaluate.py` - 评估脚本
- `inference.py` - 推理脚本
- `run_experiment.py` - 实验运行脚本

## 环境要求

- Python 3.6+
- PyTorch
- YOLOv5依赖

## 使用方法

### 1. 准备环境

```bash
# 克隆YOLOv5仓库或者下载本地YOLOv5s模型
git clone https://github.com/ultralytics/yolov5.git

# 安装依赖
pip install -r yolov5/requirements.txt
```

### 2. 运行实验

#### 完整实验流程

```bash
python run_experiment.py
```

#### 仅训练模型

```bash
python run_experiment.py --train
```

#### 仅评估模型

```bash
python run_experiment.py --evaluate
```

#### 对图像进行推理

```bash
python run_experiment.py --inference path/to/your/image.jpg
```

或者直接使用推理脚本，推理Detection>目录下的图片：

```bash
python inference.py --img path/to/your/image.jpg [--weights path/to/best.pt] [--save result.jpg]
```

## 项目运行效果

1. 数据集划分为训练集和测试集，比例为7:3或8:2
2. 计算猫和狗两个类别的检测结果，IoU阈值设为0.7
3. 统计模型的mAP指标
4. 可视化测试集上的检测结果

## 运行结果

训练完成后，可以在以下位置查看结果：

- 训练日志和权重：`pet-detection/pet-head-detection-exp/`
- 评估结果：`pet-detection/pet-head-detection-exp_evaluation/`
- 可视化结果：`pet-detection/pet-head-detection-exp/viz_results/` 