import os
import argparse
import subprocess
from pathlib import Path

def run_command(cmd):
    """运行命令并打印输出"""
    print(f"执行命令: {cmd}")
    try:
        # 使用系统默认编码，不指定编码类型
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line.strip())
        process.wait()
        return process.returncode
    except UnicodeDecodeError:
        # 如果出现编码错误，尝试使用二进制模式并手动处理输出
        print("编码错误，尝试二进制模式...")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                try:
                    print(output.decode('gbk', errors='replace').strip())  # 尝试使用GBK编码
                except:
                    print("无法解码输出")
        process.wait()
        return process.returncode

def main():
    parser = argparse.ArgumentParser(description='运行宠物头部检测实验')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--evaluate', action='store_true', help='评估模型')
    parser.add_argument('--inference', type=str, default='', help='对指定图像进行推理')
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs('pet-detection', exist_ok=True)
    
    # 准备环境
    print("\n===== 准备环境 =====")
    
    # 确保安装了必要的依赖
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
    except ImportError:
        print("安装PyTorch...")
        run_command("pip install torch")
    
    # 确保模型文件存在
    if not os.path.exists('yolov5s.pt'):
        print("下载YOLOv5预训练模型...")
        try:
            import torch
            # 使用torch.hub下载模型
            torch.hub.load('ultralytics/yolov5', 'yolov5s')
            print("预训练模型下载完成")
        except Exception as e:
            print(f"通过torch.hub下载模型失败: {e}")
            print("尝试直接下载模型文件...")
            run_command("curl -L -o yolov5s.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt")
    else:
        print("预训练模型yolov5s.pt已存在")
    
    # 如果没有指定参数，则运行完整实验
    if not (args.train or args.evaluate or args.inference):
        print("执行完整实验流程...")
        args.train = True
        args.evaluate = True
    
    # 训练模型
    if args.train:
        print("\n===== 训练模型 =====")
        run_command("python train.py")
    
    # 评估模型
    if args.evaluate:
        print("\n===== 评估模型 =====")
        run_command("python evaluate.py")
    
    # 进行推理
    if args.inference:
        print(f"\n===== 对图像进行推理: {args.inference} =====")
        run_command(f"python inference.py --img {args.inference}")
    
    print("\n实验完成!")

if __name__ == "__main__":
    main() 