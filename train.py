import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
import shutil
import logging
from datetime import datetime

# 设置日志记录，便于追踪错误
logging.basicConfig(
    filename='training_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置matplotlib，避免字体问题导致崩溃
plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]  # 使用跨平台字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class StableHelmetTrainer:
    def __init__(self, config_path="dataset/dataset.yaml", model_type="yolov8n.pt"):
        """初始化训练器，增加更多参数控制"""
        self.config_path = config_path
        self.model_type = model_type
        self.model = None
        self.results = None
        self.config = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化时记录系统信息
        self.log_system_info()
        
        try:
            # 加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # 创建必要的目录
            self.create_directories()
            
        except Exception as e:
            logging.error(f"初始化失败: {str(e)}", exc_info=True)
            raise
    
    def log_system_info(self):
        """记录系统和环境信息"""
        logging.info(f"训练开始时间: {datetime.now()}")
        logging.info(f"Python版本: {os.sys.version.split()[0]}")
        logging.info(f"PyTorch版本: {torch.__version__}")
        logging.info(f"使用设备: {self.device}")
        if self.device == "cuda":
            logging.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
            logging.info(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    def create_directories(self):
        """创建项目所需目录，增加错误处理"""
        directories = [
            "models",
            "results/inference",
            "runs/train",
            "runs/val",
            "logs"
        ]
        
        for dir_path in directories:
            try:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    logging.info(f"创建目录: {dir_path}")
            except Exception as e:
                logging.error(f"创建目录 {dir_path} 失败: {str(e)}", exc_info=True)
                raise
    
    def check_dataset(self):
        """更严格的数据集检查"""
        try:
            logging.info("开始检查数据集...")
            
            # 检查训练集
            train_images = os.listdir(self.config['train'])
            train_labels = os.listdir(os.path.join(os.path.dirname(self.config['train']), '../labels/train'))
            
            logging.info(f"训练集图片数量: {len(train_images)}")
            logging.info(f"训练集标签数量: {len(train_labels)}")
            
            # 检查验证集
            val_images = os.listdir(self.config['val'])
            val_labels = os.listdir(os.path.join(os.path.dirname(self.config['val']), '../labels/val'))
            
            logging.info(f"验证集图片数量: {len(val_images)}")
            logging.info(f"验证集标签数量: {len(val_labels)}")
            
            # 检查类别
            logging.info(f"类别数量: {self.config['nc']}")
            logging.info(f"类别名称: {self.config['names']}")
            
            # 简单的可视化，避免复杂操作导致崩溃
            self.simple_visualize_samples()
            
            # 检查是否有数据
            if len(train_images) == 0 or len(val_images) == 0:
                logging.error("训练集或验证集为空")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"数据集检查失败: {str(e)}", exc_info=True)
            return False
    
    def simple_visualize_samples(self, num_samples=2):
        """简化的可视化函数，减少崩溃风险"""
        try:
            plt.figure(figsize=(10, 5))
            
            # 获取训练集样本
            train_image_dir = self.config['train']
            train_label_dir = os.path.join(os.path.dirname(train_image_dir), '../labels/train')
            
            image_files = [f for f in os.listdir(train_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                logging.warning("训练集中没有图片文件")
                return
                
            sample_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
            
            for i, img_file in enumerate(sample_files):
                try:
                    img_path = os.path.join(train_image_dir, img_file)
                    label_path = os.path.join(train_label_dir, os.path.splitext(img_file)[0] + '.txt')
                    
                    # 读取图片，限制最大尺寸避免内存问题
                    img = cv2.imread(img_path)
                    if img is None:
                        logging.warning(f"无法读取图片: {img_path}")
                        continue
                        
                    # 缩小图片尺寸
                    scale = min(800 / img.shape[1], 600 / img.shape[0])
                    img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 简化的标签绘制
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                        
                        h, w, _ = img.shape
                        for line in lines[:5]:  # 只绘制前5个框，避免过多计算
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id, cx, cy, bw, bh = map(float, parts)
                                x1 = int((cx - bw/2) * w)
                                y1 = int((cy - bh/2) * h)
                                x2 = int((cx + bw/2) * w)
                                y2 = int((cy + bh/2) * h)
                                color = (0, 255, 0) if int(class_id) == 0 else (255, 0, 0)
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    plt.subplot(1, num_samples, i+1)
                    plt.imshow(img)
                    plt.title(f"样本 {i+1}")
                    plt.axis('off')
                
                except Exception as e:
                    logging.error(f"处理样本 {img_file} 时出错: {str(e)}")
                    continue
            
            plt.tight_layout()
            plt.savefig("results/dataset_samples.png")
            plt.close()  # 确保关闭图像，释放内存
            logging.info("数据集样本已保存至 results/dataset_samples.png")
            
        except Exception as e:
            logging.error(f"样本可视化失败: {str(e)}", exc_info=True)
            # 可视化失败不影响整体训练，仅记录日志
    
    def load_model(self):
        """加载YOLO模型，增加错误处理"""
        try:
            logging.info(f"加载模型: {self.model_type}")
            self.model = YOLO(self.model_type)
            # 移动模型到指定设备
            if self.device == "cuda":
                self.model.to("cuda")
            return self.model
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}", exc_info=True)
            raise
    
    def train(self, epochs=50, imgsz=480, batch=16, patience=10):
        """优化的训练函数，增加更多控制参数"""
        try:
            if not self.model:
                self.load_model()
                
            logging.info(f"开始训练，共{epochs}轮，批次大小{batch}，图片尺寸{imgsz}")
            
            # 训练参数优化，减少内存使用
            self.results = self.model.train(
                data=self.config_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,  # 早停耐心值
                project="runs/train",
                name="helmet_detection",
                save=True,
                device=self.device,
                workers=2,  # 减少数据加载线程
                pretrained=True,  # 使用预训练权重
                optimizer='Adam',  # 更稳定的优化器
                verbose=True
            )
            
            # 保存最佳模型
            best_model_path = os.path.join("runs/train", "helmet_detection", "weights", "best.pt")
            if os.path.exists(best_model_path):
                shutil.copy(best_model_path, os.path.join("models", "helmet_detector_best.pt"))
                logging.info("最佳模型已保存至 models/helmet_detector_best.pt")
            
            return self.results
            
        except Exception as e:
            logging.error(f"训练过程出错: {str(e)}", exc_info=True)
            raise
    
    def evaluate(self):
        """模型评估函数"""
        try:
            if not self.model:
                self.load_model()
                
            logging.info("开始评估模型...")
            metrics = self.model.val(
                data=self.config_path,
                project="runs/val",
                name="helmet_detection_eval",
                device=self.device
            )
            
            # 记录评估指标
            logging.info(f"mAP@0.5: {metrics.box.map50:.4f}")
            logging.info(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
            
            print(f"mAP@0.5: {metrics.box.map50:.4f}")
            print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
            
            return metrics
        except Exception as e:
            logging.error(f"模型评估出错: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    try:
        # 初始化训练器
        trainer = StableHelmetTrainer()
        
        # 检查数据集
        if trainer.check_dataset():
            print("数据集检查通过，开始加载模型...")
            # 加载模型
            trainer.load_model()
            
            print("开始训练...")
            # 进一步减小批次大小和图片尺寸，降低内存占用
            trainer.train(epochs=50, imgsz=480, batch=16, patience=10)
            
            print("开始评估模型...")
            trainer.evaluate()
            
            print("训练完成!")
            logging.info("训练完全部完成")
        else:
            print("数据集不完整，请检查数据!")
    except Exception as e:
        print(f"程序出错: {str(e)}")
        logging.critical(f"程序崩溃: {str(e)}", exc_info=True)
