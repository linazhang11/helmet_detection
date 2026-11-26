import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class HelmetDetector:
    def __init__(self, model_path="models/helmet_detector_best.pt", config_path="dataset/dataset.yaml"):
        """初始化检测器"""
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.class_names = []
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            self.class_names = config['names']
        
        # 加载模型
        self.load_model()
        
        # 创建结果目录
        if not os.path.exists("results/inference"):
            os.makedirs("results/inference")
    
    def load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
        print(f"加载模型: {self.model_path}")
        self.model = YOLO(self.model_path)
        return self.model
    
    def detect_image(self, image_path, save=True, show=True):
        """检测单张图片"""
        if not self.model:
            self.load_model()
            
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 进行检测
        results = self.model(img_rgb)
        
        # 处理检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 获取类别和置信度
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # 确定颜色和类别名称
                color = (0, 255, 0) if class_id == 0 else (255, 0, 0)  # 绿色: 有安全帽, 红色: 无安全帽
                class_name = self.class_names[class_id]
                
                # 绘制边界框
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
                
                # 绘制类别和置信度
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(img_rgb, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # 保存结果
        if save:
            filename = os.path.basename(image_path)
            save_path = os.path.join("results/inference", f"detected_{filename}")
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)
            print(f"检测结果已保存至: {save_path}")
        
        # 显示结果
        if show:
            plt.figure(figsize=(10, 8))
            plt.imshow(img_rgb)
            plt.title("安全帽检测结果")
            plt.axis('off')
            plt.show()
        
        return img_rgb
    
    def detect_images_in_folder(self, folder_path="data/raw"):
        """检测文件夹中的所有图片"""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
            
        # 获取所有图片文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
        
        print(f"在 {folder_path} 中找到 {len(image_files)} 张图片，开始检测...")
        
        # 逐一检测
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            print(f"检测图片: {img_path}")
            self.detect_image(img_path, save=True, show=False)
        
        print("所有图片检测完成!")

if __name__ == "__main__":
    # 初始化检测器
    detector = HelmetDetector()
    
    # 检测data/raw目录下的所有图片
    detector.detect_images_in_folder()
    
    # 单独检测一张测试图片并显示
    sample_image = "data/raw/test_image.jpg"
    if os.path.exists(sample_image):
        detector.detect_image(sample_image, save=True, show=True)
    else:
        print(f"测试图片不存在: {sample_image}")
