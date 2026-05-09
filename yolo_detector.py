import cv2
import numpy as np
from ultralytics import YOLO
from config import *


class YOLODetector:
    """
    YOLO地标粗检测器类
    作用：在复杂背景中快速锁定地标所在的矩形区域(ROI)，减少干扰并提升后期处理速度
    """

    def __init__(self, model_path=None):
        """
        初始化检测器
        参数 model_path: 模型权重文件路径，若为空则从 config 读取
        """
        self.model_path = model_path if model_path else YOLO_MODEL_PATH
        self.model = None
        self.is_loaded = False

        # 自动执行模型加载
        self._load_model()

    def _load_model(self):
        """内部函数：载入 YOLOv8/v11 模型"""
        try:
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            print(f"YOLO模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            self.is_loaded = False

    def is_valid_detection(self, box):
        """
        功能：验证 YOLO 输出的检测框是否符合物理常识
        逻辑：过滤掉面积过小或长宽比严重畸变的错误目标
        """
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        area = w * h
        ratio = w / h if h > 0 else 0

        # 1. 面积过滤：防止检测到微小噪点
        if area < YOLO_MIN_AREA:
            return False
        # 2. 形状过滤：地标为正方形，比例应接近 1:1
        if ratio < YOLO_RATIO_MIN or ratio > YOLO_RATIO_MAX:
            return False
        return True

    def detect(self, image, visualize=False):
        """
        功能：在单帧图像中执行目标检测
        参数 image: 原始 BGR 图像
        参数 visualize: 是否实时弹出窗口显示检测框
        返回: (检测结果字典列表, 选出的最优ROI坐标)
        """
        if not self.is_loaded:
            print("错误：YOLO模型尚未成功加载")
            return [], None

        # 执行推理：设置尺寸、阈值、设备等参数
        results = self.model(
            image,
            imgsz=YOLO_IMG_SIZE,
            conf=YOLO_CONF_THRES,
            iou=YOLO_IOU_THRES,
            max_det=YOLO_MAX_DET,
            device=YOLO_DEVICE,
            verbose=False  # 禁用冗余的控制台打印
        )

        detections = []
        h, w = image.shape[:2]

        # 遍历推理结果
        for r in results:
            for box, score, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())

                # 校验检测框质量
                if not self.is_valid_detection((x1, y1, x2, y2)):
                    continue

                # 记录有效目标的详细信息
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'score': float(score),
                    'class': int(cls),
                    'class_name': self.model.names[int(cls)]
                })

        best_roi = None
        if detections:
            # 策略：选取置信度最高的检测框作为后续处理的 ROI
            detections.sort(key=lambda x: x['score'], reverse=True)
            best_det = detections[0]
            x1, y1, x2, y2 = best_det['box']

            # ROI 扩展：在检测框四周增加 Padding，防止传统算法检测轮廓时边缘被切断
            padding = YOLO_ROI_PADDING
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            best_roi = (x1, y1, x2, y2)
            print(f"YOLO检测完成: 找到 {len(detections)} 个目标，最佳置信度: {best_det['score']:.2f}")
        else:
            print("YOLO提示：未在当前帧发现地标")

        # 若开启可视化，则绘制调试窗口
        if visualize:
            self.visualize_detections(image, detections, best_roi)

        return detections, best_roi

    def extract_roi(self, image, roi):
        """
        功能：根据坐标切片提取子图
        返回: (裁剪后的图像, 该子图左上角在原图中的偏移)
        """
        if roi is None:
            return image, (0, 0)

        x1, y1, x2, y2 = roi
        roi_image = image[y1:y2, x1:x2].copy()
        offset = (x1, y1)

        return roi_image, offset

    def transform_corners_to_original(self, corners, offset):
        """
        功能：坐标还原
        逻辑：将基于 ROI 局部坐标系的角点坐标，加回偏移量，映射回 1920x1080 的原图坐标系
        """
        x_offset, y_offset = offset
        transformed_corners = []
        for square_corners in corners:
            transformed_square = square_corners.copy()
            transformed_square[:, 0] += x_offset
            transformed_square[:, 1] += y_offset
            transformed_corners.append(transformed_square)

        return transformed_corners

    def visualize_detections(self, image, detections, best_roi=None):
        """内部调试函数：在图中画出 YOLO 的检测框和 ROI 范围"""
        vis_img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']}: {det['score']:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if best_roi is not None:
            x1, y1, x2, y2 = best_roi
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # 蓝色粗框代表 ROI

        cv2.imshow("YOLO Debug View", vis_img)
        cv2.waitKey(1)
        return vis_img