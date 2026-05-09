import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 绘图环境配置 ---
# 设置字体以支持中文显示，防止在可视化分析时出现乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# 解决坐标轴负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# ================== YOLO 检测参数 ==================
# YOLO模型权重的绝对路径，用于初步定位地标区域
YOLO_MODEL_PATH = r"D:\work\2-square\models\best(1).pt"
# 目标检测的置信度阈值，低于此值的检测结果将被忽略
YOLO_CONF_THRES = 0.5
# 非极大值抑制(NMS)的IOU阈值，用于合并重复的检测框
YOLO_IOU_THRES = 0.45
# YOLO网络输入的图像尺寸，模型会自动缩放图片
YOLO_IMG_SIZE = 1024
# 单张图片允许的最大检测目标数量
YOLO_MAX_DET = 10
# 指定运行设备：0代表第一块显卡，'cpu'代表处理器
YOLO_DEVICE = 0

# --- YOLO检测框合法性过滤参数 ---
# 检测框的最小面积（像素），用于过滤微小干扰
YOLO_MIN_AREA = 20
# 检测框宽高比的最小值，确保地标形状接近正方形
YOLO_RATIO_MIN = 0.8
# 检测框宽高比的最大值
YOLO_RATIO_MAX = 1.25
# 提取ROI区域时向外扩充的像素，防止切断地标边缘
YOLO_ROI_PADDING = 40

# 是否启用基于YOLO的地标预检测功能（开关）
ENABLE_YOLO_PREDETECTION = True

# ================== 真实相机标定参数 ==================
# 根据你提供的标定结果更新
CAMERA_INTRINSIC = np.array([
    [1233.645002, 0.0, 940.523440],
    [0.0, 1235.238681, 522.290987],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# 畸变系数 [k1, k2, p1, p2, k3]
DISTORTION_COEFFS = np.array([0.017707, -0.009164, -0.006018, -0.001435, 0.0], dtype=np.float32)

# ================== 传统图像处理参数 ==================
# 基础二值化阈值（在代码逻辑中主要由自适应阈值替代）
BINARY_THRESHOLD = 100
# 允许识别的最小轮廓面积
MIN_CONTOUR_AREA = 1
# 允许识别的最大轮廓面积
MAX_CONTOUR_AREA = 1000000000000
# 多边形拟合时的精度系数，越小拟合越精细
APPROX_EPSILON = 0.05
# 形态学操作（闭运算等）所使用的卷积核尺寸
MORPH_KERNEL_SIZE = 3
# 高斯模糊的核尺寸，用于降低图像噪点
GAUSSIAN_BLUR_SIZE = 3

# ================== 角点检测与精细化参数 ==================
# 亚像素角点迭代停止准则：满足精度或迭代次数即停止
CORNER_REFINEMENT_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
# 寻找角点时的搜索窗口半径大小
CORNER_WINSIZE = (5, 5)


def get_manual_landmark_points():
    """
    功能：定义地标在世界坐标系下的3D物理坐标（单位：米）
    返回：包含所有角点坐标的数组及对应的ID标签
    """
    # 定义A(大)、B(中)、C(小)三个正方形的顶点3D坐标
    manual_squares = {
        "A": [
            [0.5, -0.5, 0],   # A1: 右下
            [0.5, 0.5, 0],    # A2: 右上
            [-0.5, 0.5, 0],   # A3: 左上
            [-0.5, -0.5, 0]   # A4: 左下
        ],
        "B": [
            [0.2, -0.6, 0],   # B1
            [0.2, -0.2, 0],   # B2
            [-0.2, -0.2, 0],  # B3
            [-0.2, -0.6, 0]   # B4
        ],
        "C": [
            [0.1, -0.1, 0],   # C1
            [0.1, 0.1, 0],    # C2
            [-0.1, 0.1, 0],   # C3
            [-0.1, -0.1, 0]   # C4
        ]
    }

    all_corners = []   # 存储所有点的坐标
    square_ids = []    # 存储对应的ID
    target_order = ["A", "B", "C"] # 标准化处理顺序

    # 按照 A -> B -> C 的顺序将数据平铺到列表中
    for square_id in target_order:
        points = manual_squares[square_id]
        all_corners.extend(points)
        square_ids.extend([square_id] * 4)

    return np.array(all_corners, dtype=np.float32), square_ids


# --- 初始化地标参考数据 ---
# 获取标准模式下的3D点集
LANDMARK_3D_POINTS, SQUARE_IDS = get_manual_landmark_points()
# 仅包含B和C正方形的点集（用于近距离 small_only 模式）
LANDMARK_3D_POINTS_SMALL = LANDMARK_3D_POINTS[4:]

# ================== 单正方形解算模式的3D参考坐标 ==================
# 场景：当相机距离极近，仅能完整捕捉一个正方形时使用

# 正方形C（最小边长 0.1m）的3D点集
LANDMARK_3D_POINTS_SINGLE_C = np.array([
    [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0], [-0.1, -0.1, 0]
], dtype=np.float32)

# 正方形B（中等边长 0.2m）的3D点集
LANDMARK_3D_POINTS_SINGLE_B = np.array([
    [0.2, -0.6, 0], [0.2, -0.2, 0], [-0.2, -0.2, 0], [-0.2, -0.6, 0]
], dtype=np.float32)

# 正方形A（最大边长 1.0m）的3D点集
LANDMARK_3D_POINTS_SINGLE_A = np.array([
    [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0]
], dtype=np.float32)

# 正方形ID与实际物理尺寸的映射关系（单位：米）
SQUARE_SIZES = {
    'A': 1.0,
    'B': 0.4,
    'C': 0.2
}

# --- 启动时终端自检输出 ---
print("生成的地标角点坐标:")
print(f"总共生成 {len(LANDMARK_3D_POINTS)} 个角点")
for i, (point, sq_id) in enumerate(zip(LANDMARK_3D_POINTS, SQUARE_IDS)):
    idx_in_square = (i % 4) + 1
    print(f"{sq_id}{idx_in_square}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")

print("\n单正方形模式坐标:")
print(f"正方形C (边长0.2m): {LANDMARK_3D_POINTS_SINGLE_C.shape}")
print(f"正方形B (边长0.4m): {LANDMARK_3D_POINTS_SINGLE_B.shape}")
print(f"正方形A (边长2.0m): {LANDMARK_3D_POINTS_SINGLE_A.shape}")