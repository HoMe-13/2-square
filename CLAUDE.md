# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
python main.py          # 批量处理：逐帧检测+位姿解算+误差分析，结果输出到 visualization/pose_inference/
python 算真值.py         # 从 GNSS/IMU 原始数据生成 30fps 地面真值 CSV
python 做差.py           # 估算值与真值对比，输出轨迹图+误差分布图
python KF对比.py         # 卡尔曼滤波平滑效果可视化（需 C++ 侧先输出 KF.csv）
python csv清洗.py        # 真值数据清洗（去异常值、补零等）
python photo.py          # 从视频抽帧
python vedio.py          # 图片序列合成视频
```

## 架构

三条数据处理链：

```
视频 → photo.py → 图片序列 → main.py → 位姿估计结果
                                    ↓
GNSS/IMU 原始数据 → 算真值.py → 地面真值 CSV → 做差.py → 误差分析图
                                                         ↓
C++ KF 输出 → KF对比.py → 滤波效果图
```

**核心管线（main.py → landmark_detector.py → pose_estimator.py）：**

1. `main.py` — BatchProcessor 类，逐帧读取图片，先旋转 180°（相机倒装），调用检测器→调用位姿解算→计算相对第一帧的误差
2. `landmark_detector.py` — LandmarkDetector.detect_squares() 用红蓝通道减绿色（`R+B-3G`）增强黑色区域形态学提取大框 A，再用自适应阈值提取白框 B/C；`_sort_corners_by_topology()` 基于 C→B 向量角度确定角点排序
3. `pose_estimator.py` — PoseEstimator.estimate_pose_pnp() 根据模式（full/small_only/single_only）选择对应 3D 点集，调用 `cv2.solvePnP(flags=SOLVEPNP_IPPE)`
4. `yolo_detector.py` — YOLODetector 可选前置，检测 ROI 区域缩小搜索范围

**三个正方形的检测模式：**
- `full` — A+B+C 全检测
- `small_only` — 仅 B+C（远距离大框模糊时）
- `single_only_C/B/A` — 仅单正方形（极近/极远）

**坐标系约定：**
- 世界 3D 坐标：Z 轴朝上，正方形在 Z=0 平面，config.py 中 `get_manual_landmark_points()` 定义
- 估算位姿：第一帧作为原点，后续帧对第一帧做相对差分
- 真值对比：视觉坐标系需映射到 ENU（东-北-天），做差.py 中处理了 X↔Y 交换和符号翻转
