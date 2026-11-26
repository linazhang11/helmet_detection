import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                            QProgressBar, QMessageBox, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from ultralytics import YOLO
import torch

class DetectionThread(QThread):
    """后台检测线程，优化了检测结果处理逻辑"""
    detection_done = pyqtSignal(np.ndarray)
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, model, image=None, video_source=None):
        super().__init__()
        self.model = model
        self.image = image.copy() if image is not None else None  # 复制图片避免原始数据被修改
        self.video_source = video_source
        self.running = False
        self.conf_threshold = 0.5
        # 确保类别名称正确映射
        self.class_names = {0: 'helmet', 1: 'no_helmet'}

    def set_conf_threshold(self, threshold):
        self.conf_threshold = threshold

    def run(self):
        try:
            self.running = True
            if self.image is not None:
                self.status_updated.emit("正在检测图片...")
                self.progress_updated.emit(30)
                
                # 确保图片格式正确
                if len(self.image.shape) != 3 or self.image.shape[2] != 3:
                    raise ValueError("图片格式不正确，需要RGB或BGR格式的3通道图片")
                
                # 执行检测 - 明确指定参数
                results = self.model(
                    self.image, 
                    conf=self.conf_threshold,
                    verbose=False  # 关闭详细输出
                )
                
                self.progress_updated.emit(60)
                
                # 检查是否有检测结果
                if len(results) == 0:
                    self.status_updated.emit("未检测到任何目标")
                    self.detection_done.emit(self.image)  # 返回原始图片
                    self.progress_updated.emit(100)
                    return
                
                # 手动绘制检测结果（确保绘制逻辑可靠）
                annotated_image = self._draw_detections(results[0])
                
                self.progress_updated.emit(100)
                self.detection_done.emit(annotated_image)
                self.status_updated.emit(f"检测完成，发现{len(results[0].boxes)}个目标")
                
            elif self.video_source is not None:
                cap = cv2.VideoCapture(self.video_source)
                if not cap.isOpened():
                    self.error_occurred.emit("无法打开摄像头，请检查设备")
                    return
                
                self.status_updated.emit("正在进行实时检测...")
                
                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        self.error_occurred.emit("无法获取视频帧")
                        break
                    
                    # 执行检测
                    results = self.model(frame, conf=self.conf_threshold, verbose=False)
                    # 手动绘制检测结果
                    annotated_frame = self._draw_detections(results[0])
                    
                    self.detection_done.emit(annotated_frame)
                    # 控制帧率
                    QTimer.singleShot(30, lambda: None)  # 更稳定的延迟控制
                
                cap.release()
                self.status_updated.emit("实时检测已停止")
                
        except Exception as e:
            self.error_occurred.emit(f"检测出错: {str(e)}")
        finally:
            self.running = False

    def _draw_detections(self, result):
        """手动绘制检测框和标签，确保可视化正确"""
        # 复制原始图像
        image = result.orig_img.copy()
        h, w = image.shape[:2]
        
        # 遍历所有检测框
        for box in result.boxes:
            # 提取框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 转换为整数坐标
            # 提取类别和置信度
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 根据类别设置颜色
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # 安全帽绿色，无安全帽红色
            label = f"{self.class_names.get(cls, 'unknown')} {conf:.2f}"
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y1_label = max(y1, label_size[1] + 10)
            cv2.rectangle(
                image, 
                (x1, y1_label - label_size[1] - 10),
                (x1 + label_size[0], y1_label + base_line - 10),
                color, 
                -1
            )
            
            # 绘制标签文本
            cv2.putText(
                image, 
                label, 
                (x1, y1_label - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
        
        return image

    def stop(self):
        self.running = False
        self.wait()

class HelmetDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.detection_thread = None
        self.current_image = None  # 保存当前处理的原始图像
        self.init_ui()
        
    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("安全帽检测系统")
        self.setGeometry(100, 100, 1000, 700)
        self.setMinimumSize(800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 1. 模型加载区域
        model_layout = QHBoxLayout()
        self.model_path_label = QLabel("未加载模型")
        self.model_path_label.setMinimumWidth(400)
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        self.device_label = QLabel("设备: CPU")
        
        model_layout.addWidget(QLabel("模型状态:"))
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(self.load_model_btn)
        model_layout.addWidget(self.device_label)
        model_layout.addStretch()
        
        # 2. 检测控制区域
        control_layout = QHBoxLayout()
        self.select_image_btn = QPushButton("选择图片检测")
        self.select_image_btn.clicked.connect(self.select_image)
        self.select_image_btn.setEnabled(False)
        
        self.camera_detect_btn = QPushButton("摄像头实时检测")
        self.camera_detect_btn.clicked.connect(self.start_camera_detection)
        self.camera_detect_btn.setEnabled(False)
        
        self.stop_detect_btn = QPushButton("停止检测")
        self.stop_detect_btn.clicked.connect(self.stop_detection)
        self.stop_detect_btn.setEnabled(False)
        
        # 置信度阈值控制
        self.conf_label = QLabel("置信度阈值: 0.5")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 9)
        self.conf_slider.setValue(5)  # 0.5
        self.conf_slider.valueChanged.connect(self.update_conf_threshold)
        
        control_layout.addWidget(self.select_image_btn)
        control_layout.addWidget(self.camera_detect_btn)
        control_layout.addWidget(self.stop_detect_btn)
        control_layout.addWidget(self.conf_label)
        control_layout.addWidget(self.conf_slider)
        control_layout.addStretch()
        
        # 3. 进度和状态区域
        status_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: blue;")
        
        status_layout.addWidget(QLabel("进度:"))
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        # 4. 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #cccccc;")
        self.image_label.setText("请加载模型并选择图片或启动摄像头")
        
        # 5. 底部信息区域
        info_layout = QHBoxLayout()
        self.info_label = QLabel("提示: 先加载训练好的模型文件(best.pt)，然后选择检测方式")
        self.info_label.setStyleSheet("color: #666666; font-size: 12px;")
        info_layout.addWidget(self.info_label)
        
        # 组装主布局
        main_layout.addLayout(model_layout)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(status_layout)
        main_layout.addWidget(self.image_label, 1)
        main_layout.addLayout(info_layout)
        
        # 初始化变量
        self.conf_threshold = 0.5

    def load_model(self):
        """加载YOLO模型并验证有效性"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt);;所有文件 (*)"
        )
        
        if not model_path:
            return
            
        try:
            self.status_label.setText("正在加载模型...")
            self.status_label.setStyleSheet("color: orange;")
            QApplication.processEvents()
            
            # 加载模型并验证
            self.model = YOLO(model_path)
            
            # 简单测试模型是否有效
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_results = self.model(test_img, conf=0.1, verbose=False)
            if len(test_results) == 0:
                raise ValueError("模型加载成功但无法进行推理，请检查模型文件是否损坏")
            
            # 检查设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                self.model.to("cuda")
                self.device_label.setText(f"设备: GPU ({torch.cuda.get_device_name(0)})")
            else:
                self.device_label.setText("设备: CPU")
            
            # 更新UI
            self.model_path_label.setText(os.path.basename(model_path))
            self.status_label.setText("模型加载成功")
            self.status_label.setStyleSheet("color: green;")
            self.select_image_btn.setEnabled(True)
            self.camera_detect_btn.setEnabled(True)
            
        except Exception as e:
            self.model = None
            self.status_label.setText(f"模型加载失败: {str(e)}")
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "错误", f"加载模型时出错:\n{str(e)}")

    def select_image(self):
        """选择图片并确保正确处理"""
        self.stop_detection()
        
        image_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        
        if not image_path:
            return
            
        try:
            # 读取图片并验证
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise Exception("无法读取图片文件，请检查文件是否损坏")
            
            # 显示原始图片
            self.display_image(self.current_image)
            
            # 启动检测线程
            self.detection_thread = DetectionThread(self.model, image=self.current_image)
            self.detection_thread.detection_done.connect(self.on_detection_done)
            self.detection_thread.progress_updated.connect(self.progress_bar.setValue)
            self.detection_thread.status_updated.connect(self.update_status)
            self.detection_thread.error_occurred.connect(self.show_error)
            self.detection_thread.set_conf_threshold(self.conf_threshold)
            self.detection_thread.start()
            
            # 更新按钮状态
            self.stop_detect_btn.setEnabled(True)
            self.select_image_btn.setEnabled(False)
            self.camera_detect_btn.setEnabled(False)
            
        except Exception as e:
            self.status_label.setText(f"处理图片出错: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    def start_camera_detection(self):
        """启动摄像头检测"""
        self.stop_detection()
        
        try:
            self.detection_thread = DetectionThread(self.model, video_source=0)
            self.detection_thread.detection_done.connect(self.on_detection_done)
            self.detection_thread.status_updated.connect(self.update_status)
            self.detection_thread.error_occurred.connect(self.show_error)
            self.detection_thread.set_conf_threshold(self.conf_threshold)
            self.detection_thread.start()
            
            self.stop_detect_btn.setEnabled(True)
            self.select_image_btn.setEnabled(False)
            self.camera_detect_btn.setEnabled(False)
            
        except Exception as e:
            self.status_label.setText(f"启动摄像头出错: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    def stop_detection(self):
        """停止当前检测"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread = None
        
        # 恢复按钮状态
        self.stop_detect_btn.setEnabled(False)
        if self.model:
            self.select_image_btn.setEnabled(True)
            self.camera_detect_btn.setEnabled(True)
        
        self.progress_bar.setValue(0)

    def on_detection_done(self, image):
        """处理检测结果并显示"""
        self.display_image(image)
        
        # 如果是图片检测，恢复按钮状态
        if self.current_image is not None and (not self.detection_thread or not self.detection_thread.isRunning()):
            self.select_image_btn.setEnabled(True)
            self.camera_detect_btn.setEnabled(True)
            self.stop_detect_btn.setEnabled(False)

    def display_image(self, cv_image):
        """改进的图像显示逻辑，确保正确转换格式"""
        try:
            # 检查图像是否有效
            if cv_image is None or not isinstance(cv_image, np.ndarray):
                raise ValueError("无效的图像数据")
                
            # 确保图像是3通道
            if len(cv_image.shape) == 2:  # 灰度图转换为RGB
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            elif cv_image.shape[2] == 4:  # RGBA转换为RGB
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2RGB)
            else:  # BGR转换为RGB
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            h, w, ch = cv_image.shape
            bytes_per_line = ch * w
            
            # 转换为QImage
            q_image = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 缩放图片保持比例
            scaled_q_image = q_image.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(QPixmap.fromImage(scaled_q_image))
            
        except Exception as e:
            self.show_error(f"显示图像出错: {str(e)}")
            self.image_label.setText(f"无法显示图像: {str(e)}")

    def update_status(self, message):
        """更新状态信息"""
        self.status_label.setText(message)
        
        if "成功" in message or "完成" in message:
            self.status_label.setStyleSheet("color: green;")
        elif "正在" in message:
            self.status_label.setStyleSheet("color: orange;")
        elif "未检测到" in message:
            self.status_label.setStyleSheet("color: #FFA500;")  # 橙色
        else:
            self.status_label.setStyleSheet("color: blue;")

    def show_error(self, message):
        """显示错误信息"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: red;")
        QMessageBox.warning(self, "操作提示", message)

    def update_conf_threshold(self):
        """更新置信度阈值"""
        value = self.conf_slider.value() / 10.0
        self.conf_threshold = value
        self.conf_label.setText(f"置信度阈值: {value:.1f}")
        
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.set_conf_threshold(value)

    def resizeEvent(self, event):
        """窗口大小改变时重新调整图片"""
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.display_image(self.current_image)
        super().resizeEvent(event)

    def closeEvent(self, event):
        """关闭窗口时停止所有线程"""
        self.stop_detection()
        event.accept()

if __name__ == "__main__":
    # 确保中文显示正常
    import matplotlib
    matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
    
    app = QApplication(sys.argv)
    window = HelmetDetectionGUI()
    window.show()
    sys.exit(app.exec_())
    