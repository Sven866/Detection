import torch
import yaml
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import os

# 加载配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

def inference(img_path, weights_path=None):
    """
    使用训练好的模型进行推理
    
    Args:
        img_path: 输入图像路径
        weights_path: 模型权重路径，如果为None则使用训练的最佳模型
    
    Returns:
        results: 推理结果
    """
    if weights_path is None:
        # 使用训练的最佳模型
        weights_path = f"{config['project']}/{config['name']}/weights/best.pt"
    
    # 检查模型是否存在
    if not os.path.exists(weights_path):
        print(f"未找到模型权重文件: {weights_path}")
        print("尝试查找最后保存的模型...")
        
        last_weights_path = f"{config['project']}/{config['name']}/weights/last.pt"
        if os.path.exists(last_weights_path):
            print(f"找到最后保存的模型: {last_weights_path}")
            weights_path = last_weights_path
        else:
            print("未找到任何训练好的模型，使用本地预训练模型yolov5s.pt")
            if os.path.exists('yolov5s.pt'):
                weights_path = 'yolov5s.pt'
            else:
                print("错误：未找到yolov5s.pt，无法进行推理")
                return None
    
    print(f"使用模型: {weights_path}")
    
    # 直接使用torch.hub加载本地模型，但不通过hub下载
    try:
        # 为避免使用hub下载，我们使用force_reload=False
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.abspath(weights_path), force_reload=False)
        
        # 设置置信度阈值
        model.conf = config['conf_thres']
        model.iou = config['iou_thres']
        
        # 执行推理
        img_path_abs = os.path.abspath(img_path) if img_path else ""
        print(f"执行推理，图像路径: {img_path_abs}")
        if not os.path.exists(img_path_abs):
            print(f"警告：图像文件不存在: {img_path_abs}")
        results = model(img_path_abs)
        
        return results
    
    except Exception as e:
        print(f"推理过程中出错: {e}")
        raise e

def visualize_result(results, img_path, save_path=None):
    """可视化推理结果"""
    # 渲染结果
    rendered_img = results.render()[0]
    
    # 显示结果
    plt.figure(figsize=(12, 8))
    plt.imshow(rendered_img)
    plt.axis('off')
    plt.title('检测结果')
    
    # 保存结果
    if save_path:
        plt.savefig(save_path)
        print(f"结果已保存到: {save_path}")
    
    # 显示详细信息
    df = results.pandas().xyxy[0]
    print("\n检测结果详情:")
    print(df)
    
    return rendered_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv5宠物头部检测推理')
    parser.add_argument('--img', type=str, required=True, help='输入图像路径')
    parser.add_argument('--weights', type=str, default=None, help='模型权重路径')
    parser.add_argument('--save', type=str, default='detection_result.jpg', help='保存结果路径')
    args = parser.parse_args()
    
    # 执行推理
    results = inference(args.img, args.weights)
    
    # 可视化结果
    visualize_result(results, args.img, args.save)
    
    # 打印结果
    results.print()  # 或 .show(), .save(), .crop(), .pandas() 等