import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 加载配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def evaluate_model():
    print("开始评估模型...")
    
    # 加载最佳模型
    weights_path = f"{config['project']}/{config['name']}/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"未找到模型权重文件: {weights_path}")
        print("尝试查找最后保存的模型...")
        
        last_weights_path = f"{config['project']}/{config['name']}/weights/last.pt"
        if os.path.exists(last_weights_path):
            print(f"找到最后保存的模型: {last_weights_path}")
            weights_path = last_weights_path
        else:
            print("未找到任何训练好的模型，使用本地预训练模型yolov5s.pt进行评估")
            if os.path.exists('yolov5s.pt'):
                weights_path = 'yolov5s.pt'
            else:
                print("错误：未找到yolov5s.pt，无法进行评估")
                return
    
    print(f"使用模型: {weights_path}")
    
    # 创建临时数据集YAML文件
    temp_yaml_path = 'temp_eval_dataset.yaml'
    
    # 获取当前工作目录的绝对路径
    current_dir = os.path.abspath('.')
    
    # 构建数据集路径（使用绝对路径避免路径问题）
    train_path = os.path.join(current_dir, config['train'])
    val_path = os.path.join(current_dir, config['val'])
    test_path = os.path.join(current_dir, config['test'])
    
    dataset_config = {
        'path': current_dir,  # 使用当前工作目录作为基础路径
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': config['nc'],
        'names': config['names']
    }
    
    print(f"创建评估数据集配置: {dataset_config}")
    
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f)

    # 如果使用的是预训练模型yolov5s.pt，它有80个类别，和我们的数据集(2个类别)不匹配
    # 这种情况下不能用val.py进行评估，而是需要用detect.py进行简单推理
    if os.path.basename(weights_path) == 'yolov5s.pt':
        print("使用预训练模型进行推理而不是评估...")
        cmd = (f"python -m yolov5.detect --source {os.path.abspath(test_path)} "
              f"--weights {os.path.abspath(weights_path)} "
              f"--conf {config['conf_thres']} "
              f"--iou {config['iou_thres']} "
              f"--save-txt "
              f"--save-conf "
              f"--project {config['project']} "
              f"--name {config['name']}_detection")
        
        print(f"执行命令: {cmd}")
        os.system(cmd)
        
        print(f"推理结果保存在 {config['project']}/{config['name']}_detection")
        return
    
    # 直接使用命令行方式进行评估
    print("使用命令行方式评估YOLOv5模型...")
    cmd = (f"python -m yolov5.val --data {os.path.abspath(temp_yaml_path)} "
          f"--weights {os.path.abspath(weights_path)} "
          f"--batch-size {config['batch_size']} "
          f"--img-size {config['img_size']} "
          f"--task test "
          f"--conf-thres {config['conf_thres']} "
          f"--iou-thres {config['iou_thres']} "
          f"--save-json "
          f"--save-txt "
          f"--save-conf "
          f"--project {config['project']} "
          f"--name {config['name']}_evaluation")
    
    print(f"执行命令: {cmd}")
    os.system(cmd)

def visualize_results(num_samples=3):
    print("可视化测试结果...")
    
    # 加载最佳模型
    weights_path = f"{config['project']}/{config['name']}/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"未找到模型权重文件: {weights_path}")
        print("尝试查找最后保存的模型...")
        
        last_weights_path = f"{config['project']}/{config['name']}/weights/last.pt"
        if os.path.exists(last_weights_path):
            print(f"找到最后保存的模型: {last_weights_path}")
            weights_path = last_weights_path
        else:
            print("未找到任何训练好的模型，使用本地预训练模型yolov5s.pt进行可视化")
            if os.path.exists('yolov5s.pt'):
                weights_path = 'yolov5s.pt'
            else:
                print("错误：未找到yolov5s.pt，无法进行可视化")
                return
    
    print(f"使用模型: {weights_path}")
    
    # 加载模型
    print("加载模型...")
    try:
        # 使用force_reload=False避免重新下载
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.abspath(weights_path), force_reload=False)
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 设置置信度阈值
    model.conf = config['conf_thres']
    model.iou = config['iou_thres']
    
    # 获取测试集图像
    test_path = Path(os.path.abspath(config['test']))
    print(f"查找测试集图像: {test_path}")
    test_images = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
    
    if not test_images:
        print(f"未找到测试图像，路径: {test_path}")
        # 尝试查找子目录中的图像
        for subdir in test_path.glob('*'):
            if subdir.is_dir():
                print(f"检查子目录: {subdir}")
                test_images.extend(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
        
        if not test_images:
            # 尝试使用训练集目录作为替代
            if os.path.exists(os.path.join(os.path.dirname(test_path), 'train')):
                train_path = Path(os.path.join(os.path.dirname(test_path), 'train'))
                print(f"使用训练集目录作为替代: {train_path}")
                for subdir in train_path.glob('*'):
                    if subdir.is_dir():
                        print(f"检查子目录: {subdir}")
                        test_images.extend(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
                test_images.extend(list(train_path.glob('*.jpg')) + list(train_path.glob('*.png')))
                print(f"在训练集中找到图像: {len(test_images)}")
        
        if not test_images:
            print("无法找到任何图像进行可视化")
            return
    
    print(f"找到{len(test_images)}张测试图像")
    
    # 创建结果目录
    results_dir = Path(f"{config['project']}/{config['name']}/viz_results")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # 随机选择几张图片
    selected_images = np.random.choice(test_images, min(num_samples, len(test_images)), replace=False)
    
    # 对每张图片进行推理并保存结果
    fig, axes = plt.subplots(1, len(selected_images), figsize=(6*len(selected_images), 6))
    if len(selected_images) == 1:
        axes = [axes]
    
    for i, img_path in enumerate(selected_images):
        # 执行推理
        print(f"对图像进行推理: {img_path}")
        try:
            results = model(str(img_path))
            
            # 保存带有标注的图像
            res_img = results.render()[0]
            
            # 显示结果
            axes[i].imshow(res_img)
            axes[i].set_title(f'测试图像 {i+1}')
            axes[i].axis('off')
            
            # 保存单张结果
            img_save_path = results_dir / f"test_result_{i+1}.jpg"
            plt.imsave(str(img_save_path), res_img)
        except Exception as e:
            print(f"处理图像时出错: {e}")
            axes[i].text(0.5, 0.5, '推理失败', ha='center', va='center')
            axes[i].axis('off')
    
    # 保存整体可视化
    plt.tight_layout()
    plt.savefig(str(results_dir / "test_samples.jpg"))
    plt.close()
    
    print(f"可视化结果保存在 {results_dir}")
    
    return results_dir

if __name__ == "__main__":
    # 评估模型
    evaluate_model()
    
    # 可视化结果
    visualize_results() 