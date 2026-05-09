# 2-square — 地标视觉定位系统

基于 YOLO + PnP 的视觉地标位姿估计项目，用于无人机/移动平台的空间定位。

## 地标设计

三个嵌套正方形（A/B/C），物理尺寸：

| 正方形 | 边长 | 位置 |
|--------|------|------|
| A（大黑框） | 1.0m | 最外层 |
| B（中白框） | 0.4m | 内层偏下 |
| C（小白框） | 0.2m | 中间偏下 |

## 核心流程

1. **YOLO 粗定位** — 检测地标 ROI，排除背景干扰
2. **传统视觉检测** — 自适应阈值 + 轮廓拟合 + 亚像素角点
3. **PnP 位姿解算** — solvePnP 计算相机三维姿态
4. **误差分析** — 对比地面真值（GNSS+IMU）输出 2D 位置误差

## 主要文件

| 文件 | 功能 |
|------|------|
| `config.py` | 相机内参、地标 3D 坐标、YOLO 参数 |
| `main.py` | 批量处理入口，逐帧解算 + 可视化 |
| `yolo_detector.py` | YOLO 目标检测 + ROI 提取 |
| `landmark_detector.py` | 正方形检测 + 角点排序 |
| `pose_estimator.py` | PnP 解算 + 坐标轴绘制 |
| `photo.py` | 视频抽帧工具 |
| `vedio.py` | 图片合成为视频 |
| `算真值.py` | 从 GNSS/IMU 数据生成地面真值 |
| `做差.py` | 估算值与真值对比分析 |
| `KF对比.py` | 卡尔曼滤波效果可视化 |
| `csv清洗.py` | 真值数据清洗预处理 |

## 运行

```bash
python main.py
```

结果输出至 `visualization/pose_inference/`，包含标注图、二值化图和误差统计 CSV。

## 环境要求

- Python 3.10+
- ultralytics (YOLO)
- OpenCV
- NumPy / Pandas / Matplotlib
- pymap3d, scipy
