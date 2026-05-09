import cv2
import numpy as np
import math
from config import *


class PoseEstimator:
    """
    位姿估计器类：实现从2D图像坐标到3D空间位姿的变换计算
    """

    def __init__(self):
        # 预载配置中的相机与3D点参考数据
        self.camera_matrix = CAMERA_INTRINSIC
        self.dist_coeffs = DISTORTION_COEFFS
        self.object_points = LANDMARK_3D_POINTS
        self.object_points_small = LANDMARK_3D_POINTS_SMALL
        self.object_points_single_A = LANDMARK_3D_POINTS_SINGLE_A
        self.object_points_single_B = LANDMARK_3D_POINTS_SINGLE_B
        self.object_points_single_C = LANDMARK_3D_POINTS_SINGLE_C

    def estimate_pose_pnp(self, image_points, mode):
        """
        功能：调用 OpenCV 的 PnP 算法解算位姿
        参数 image_points: 排序后的2D角点列表
        参数 mode: 当前检测模式
        """
        # 1. 根据模式选择匹配的3D世界坐标点
        if mode == "full":
            obj = self.object_points
        elif mode == "small_only":
            obj = self.object_points_small
        elif mode == "single_only_A":
            obj = self.object_points_single_A
        elif mode == "single_only_B":
            obj = self.object_points_single_B
        elif mode == "single_only_C":
            obj = self.object_points_single_C
        else:
            return None, None

        # 校验：2D点数必须与3D点数一一对应
        if len(image_points) != len(obj):
            print("角点数量不匹配！")
            return None, None

        try:
            # 2. 调用 solvePnP 核心函数
            # 对于单正方形(4个点)，SOLVEPNP_IPPE 是目前平面位姿估计中最精准的算法
            success, rvec, tvec = cv2.solvePnP(
                obj,
                np.array(image_points, dtype=np.float32),
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE
            )

            if success:
                print(f"PnP位姿估计成功 (模式: {mode})")
                return rvec, tvec
            else:
                return None, None

        except Exception as e:
            print(f"PnP解算异常: {e}")
            return None, None

    def rotation_matrix_to_euler_angles(self, R):
        """
        功能：将3x3旋转矩阵转换为 Roll, Pitch, Yaw 欧拉角（单位：度）
        公式：利用 atan2 根据矩阵元素反算旋转角度
        """
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6  # 判断是否处于万向锁临界点

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])  # Roll
            y = math.atan2(-R[2, 0], sy)  # Pitch
            z = math.atan2(R[1, 0], R[0, 0])  # Yaw
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.rad2deg(np.array([x, y, z]))

    def print_pose_info(self, rvec, tvec, method="Unknown"):
        """输出位姿详细文本信息到控制台"""
        if rvec is None or tvec is None: return
        print(f"\n=== {method}位姿估计结果 ===")
        print(f"平移向量 (T): {tvec.flatten()}")
        print(f"旋转向量 (R): {rvec.flatten()}")

    def draw_coordinate_axes(self, image, rvec, tvec, length=1):
        """在图像上绘制代表世界坐标系的红(X)绿(Y)蓝(Z)三轴"""
        if rvec is None or tvec is None: return image
        result_img = image.copy()
        cv2.drawFrameAxes(result_img, self.camera_matrix, self.dist_coeffs,
                          rvec, tvec, length, thickness=3)
        return result_img

    def draw_detection_results(self, image, corners, square_names=None):
        """在图像上标注检测到的正方形、角点序号和ID名称"""
        result_img = image.copy()
        # 角点标注颜色序列：绿、蓝、红、黄
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

        if square_names is None: square_names = ['A', 'B', 'C']

        for i, (square_corners, square_name) in enumerate(zip(corners, square_names)):
            for j, corner in enumerate(square_corners):
                # 绘制实心小圆点
                cv2.circle(result_img, tuple(corner.astype(int)), 5, colors[j], -1)
                # 标注点索引（如A.0）
                cv2.putText(result_img, f'{square_name}.{j}',
                            tuple(corner.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 绘制正方形闭合轮廓
            cv2.polylines(result_img, [square_corners.astype(int)], True,
                          (0, 255, 255) if square_name.startswith('B') else (255, 0, 255), 2)

            # 在中心位置标注正方形ID
            center = np.mean(square_corners, axis=0).astype(int)
            cv2.putText(result_img, square_name,
                        tuple(center + np.array([10, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return result_img