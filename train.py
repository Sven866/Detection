import os
import yaml
import torch
import random
import numpy as np
from pathlib import Path

# 设置随机种子，确保结果可复现
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# 加载配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 准备训练命令
def train():
    print("开始训练YOLOv5模型...")
    
    # 创建临时数据集YAML文件
    temp_yaml_path = 'temp_dataset.yaml'
    
    # 获取当前工作目录的绝对路径
    current_dir = os.path.abspath('.')
    
    # 构建数据集路径（使用绝对路径避免路径问题）
    train_path = os.path.join(current_dir, config['train'])
    val_path = os.path.join(current_dir, config['val'])
    test_path = os.path.join(current_dir, config['test'])
    
    # 检查路径是否存在
    print(f"检查训练集路径: {train_path}")
    print(f"路径存在: {os.path.exists(train_path)}")
    
    dataset_config = {
        'path': current_dir,  # 使用当前工作目录作为基础路径
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': config['nc'],
        'names': config['names']
    }
    
    print(f"创建数据集配置: {dataset_config}")
    
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f)
    
    # 直接使用命令行方式运行YOLOv5训练
    # yolov5包不支持直接训练，只支持推理
    print("使用命令行方式训练YOLOv5模型...")
    cmd = (f"python -m yolov5.train --img {config['img_size']} "
           f"--batch {config['batch_size']} "
           f"--epochs {config['epochs']} "
           f"--data {os.path.abspath(temp_yaml_path)} "  # 使用绝对路径
           f"--weights {os.path.abspath('yolov5s.pt')} "  # 使用绝对路径
           f"--project {config['project']} "
           f"--name {config['name']} "
           f"--workers {config['workers']}")
    
    print(f"执行命令: {cmd}")
    os.system(cmd)

# 统计训练集和测试集数据量
def dataset_stats():
    # 计算训练集和测试集的图像数量
    train_path = Path(config['train'])
    val_path = Path(config['val'])
    
    train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
    val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
    
    total_images = len(train_images) + len(val_images)
    train_ratio = len(train_images) / total_images
    val_ratio = len(val_images) / total_images
    
    print(f"训练集图像数量: {len(train_images)}")
    print(f"测试集图像数量: {len(val_images)}")
    print(f"总图像数量: {total_images}")
    print(f"训练集比例: {train_ratio:.2f}")
    print(f"测试集比例: {val_ratio:.2f}")
    
    # 统计类别分布
    train_cat = 0
    train_dog = 0
    val_cat = 0
    val_dog = 0
    
    # 统计训练集中猫和狗的数量
    for img_path in train_images:
        label_path = str(img_path).replace('images', 'labels').replace(img_path.suffix, '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if class_id == 0:  # 猫
                        train_cat += 1
                    elif class_id == 1:  # 狗
                        train_dog += 1
    
    # 统计验证集中猫和狗的数量
    for img_path in val_images:
        label_path = str(img_path).replace('images', 'labels').replace(img_path.suffix, '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if class_id == 0:  # 猫
                        val_cat += 1
                    elif class_id == 1:  # 狗
                        val_dog += 1
    
    print(f"训练集中猫的数量: {train_cat}")
    print(f"训练集中狗的数量: {train_dog}")
    print(f"测试集中猫的数量: {val_cat}")
    print(f"测试集中狗的数量: {val_dog}")

if __name__ == "__main__":
    # 打印数据集统计信息
    dataset_stats()
    
    # 开始训练
    train()