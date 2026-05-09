import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# ⚙️ 第一步：配置参数区
# ==========================================
GT_FILE = 'Video_Sync_30fps_GroundTruth_Zeroed.csv'  # 你的真值 CSV 文件名
EST_FILE = 'estimate.csv'  # 你用 C++ 跑出来的估算值 CSV 文件名

FRAME_COL = 'Video_Frame'  # 对齐基准列（帧号）

# 设置图表全局样式 (多版本自适应，解决之前的报错)
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')

sns.set_context("notebook", font_scale=1.1)


# ==========================================
# 🧮 核心算法辅助函数
# ==========================================
def calc_angle_diff(est, gt):
    """
    计算两个角度（度）的最短差值，处理 360 度循环跃变问题
    返回范围: [-180, 180]
    """
    diff = est - gt
    return (diff + 180) % 360 - 180


# ==========================================
# 🚀 主分析流程
# ==========================================
def main():
    print("🚀 [1/6] 正在读取数据文件...")
    if not os.path.exists(GT_FILE) or not os.path.exists(EST_FILE):
        print(f"❌ 错误: 找不到文件 {GT_FILE} 或 {EST_FILE}，请检查路径。")
        return

    df_gt = pd.read_csv(GT_FILE)
    df_est = pd.read_csv(EST_FILE)

    print("🔄 [2/6] 正在通过帧号 (Video_Frame) 进行精准对齐...")
    df_merged = pd.merge(df_gt, df_est, on=FRAME_COL, suffixes=('_gt', '_est'))
    total_frames = len(df_merged)

    # ---------------------------------------------------------
    # 🌟 核心优化 1：过滤空值/无效帧 (剔除全为 0.0000 的丢失帧)
    # ---------------------------------------------------------
    print("🧹 [3/6] 正在清洗数据，剔除目标丢失的无效帧...")
    # 判断条件：如果平移的 X, Y, Z 全部严格等于 0.0，判定为算法丢失帧
    valid_mask = ~((df_merged['est_x_east(m)'] == 0.0) &
                   (df_merged['est_y_north(m)'] == 0.0) &
                   (df_merged['est_z_up(m)'] == 0.0))

    df_valid = df_merged[valid_mask].copy()
    valid_frames = len(df_valid)
    print(
        f"   => 原始匹配: {total_frames} 帧 | 有效解算: {valid_frames} 帧 | 丢失剔除: {total_frames - valid_frames} 帧")

    if valid_frames == 0:
        print("❌ 错误: 剔除全 0 帧后，没有剩余的有效数据，无法继续分析！")
        return

    # ---------------------------------------------------------
    # 🌟 核心优化 2：三维空间刚体变换 (Camera -> ENU)
    # ---------------------------------------------------------
    print("📐 [4/6] 正在执行三维空间坐标系映射 (视觉 -> 东北天)...")
    # 平移映射：ENU X(东)=-Camera Y(西) | ENU Y(北)=-Camera X(南) | ENU Z(上)=-Camera Z(下)
    df_valid['real_est_x_enu'] = df_valid['est_y_north(m)']
    df_valid['real_est_y_enu'] = df_valid['est_x_east(m)']
    df_valid['real_est_z_enu'] = df_valid['est_z_up(m)']

    # 旋转映射：轴向翻转导致旋转符号取反，且 Roll 和 Pitch 含义对调
    df_valid['real_est_roll_enu'] = -df_valid['est_pitch(deg)']
    df_valid['real_est_pitch_enu'] = -df_valid['est_roll(deg)']
    df_valid['real_est_yaw_enu'] = -df_valid['est_yaw(deg)']

    # 定义映射字典，指向我们刚刚修正后的真实 ENU 估算数据
    COLUMNS_MAP = {
        'x': ('rel_x_east(m)', 'real_est_x_enu'),
        'y': ('rel_y_north(m)', 'real_est_y_enu'),
        'z': ('rel_z_up(m)', 'real_est_z_enu'),
        'roll': ('rel_roll(deg)', 'real_est_roll_enu'),
        'pitch': ('rel_pitch(deg)', 'real_est_pitch_enu'),
        'yaw': ('rel_yaw(deg)', 'real_est_yaw_enu')
    }

    # ---------------------------------------------------------
    # 计算误差
    # ---------------------------------------------------------
    print("🧮 [5/6] 正在计算各项相对误差与绝对误差...")
    for axis in ['x', 'y', 'z']:
        gt_col, est_col = COLUMNS_MAP[axis]
        df_valid[f'err_{axis}'] = df_valid[est_col] - df_valid[gt_col]
        df_valid[f'abs_err_{axis}'] = df_valid[f'err_{axis}'].abs()

    for axis in ['roll', 'pitch', 'yaw']:
        gt_col, est_col = COLUMNS_MAP[axis]
        df_valid[f'err_{axis}'] = calc_angle_diff(df_valid[est_col], df_valid[gt_col])
        df_valid[f'abs_err_{axis}'] = df_valid[f'err_{axis}'].abs()

    df_valid['err_distance_2d'] = np.sqrt(df_valid['err_x'] ** 2 + df_valid['err_y'] ** 2)

    # 打印最终统计摘要
    print("\n" + "=" * 55)
    print("📊 核心精度统计摘要 (Mean ± Std | Max) - 剔除无效帧后")
    print("=" * 55)
    metrics = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'distance_2d']
    units = {'x': 'm', 'y': 'm', 'z': 'm', 'roll': 'deg', 'pitch': 'deg', 'yaw': 'deg', 'distance_2d': 'm'}

    for m in metrics:
        err_col = f'abs_err_{m}' if m != 'distance_2d' else 'err_distance_2d'
        col_mean = df_valid[err_col].mean()
        col_std = df_valid[err_col].std()
        col_max = df_valid[err_col].max()
        print(f"{m.upper():<12}: {col_mean:>7.4f} ± {col_std:>7.4f} {units[m]:<3} | Max: {col_max:>7.4f} {units[m]}")
    print("=" * 55 + "\n")

    # ---------------------------------------------------------
    # 可视化出图
    # ---------------------------------------------------------
    print("📈 [6/6] 正在生成可视化分析图表...")

    # 1. 2D 轨迹对比图
    plt.figure(figsize=(10, 8))
    plt.plot(df_valid[COLUMNS_MAP['x'][0]], df_valid[COLUMNS_MAP['y'][0]],
             label='Ground Truth (ENU)', color='black', linewidth=2, linestyle='--')
    plt.plot(df_valid[COLUMNS_MAP['x'][1]], df_valid[COLUMNS_MAP['y'][1]],
             label='Estimated Pose (Vision Aligned)', color='red', linewidth=1.5, alpha=0.8)

    plt.scatter(df_valid[COLUMNS_MAP['x'][0]].iloc[0], df_valid[COLUMNS_MAP['y'][0]].iloc[0], c='green', s=100,
                label='Start', zorder=5)
    plt.scatter(df_valid[COLUMNS_MAP['x'][0]].iloc[-1], df_valid[COLUMNS_MAP['y'][0]].iloc[-1], c='blue', s=100,
                label='End', marker='X', zorder=5)

    plt.title('2D Trajectory Comparison (X-East, Y-North)', fontsize=14, fontweight='bold')
    plt.xlabel('X Easting (m)')
    plt.ylabel('Y Northing (m)')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('analysis_1_trajectory_aligned.png', dpi=300)

    # 2. 逐帧误差曲线图
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle('Error vs Video Frame (Aligned & Filtered)', fontsize=18, fontweight='bold')

    frames = df_valid[FRAME_COL]

    trans_axes = ['x', 'y', 'z']
    colors_trans = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, axis in enumerate(trans_axes):
        axes[i, 0].plot(frames, df_valid[f'err_{axis}'], color=colors_trans[i], linewidth=1)
        axes[i, 0].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[i, 0].set_ylabel(f'{axis.upper()} Error (m)', fontweight='bold')
        axes[i, 0].fill_between(frames, 0, df_valid[f'err_{axis}'], color=colors_trans[i], alpha=0.2)

    rot_axes = ['roll', 'pitch', 'yaw']
    colors_rot = ['#9467bd', '#8c564b', '#e377c2']
    for i, axis in enumerate(rot_axes):
        axes[i, 1].plot(frames, df_valid[f'err_{axis}'], color=colors_rot[i], linewidth=1)
        axes[i, 1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[i, 1].set_ylabel(f'{axis.capitalize()} Error (deg)', fontweight='bold')
        axes[i, 1].fill_between(frames, 0, df_valid[f'err_{axis}'], color=colors_rot[i], alpha=0.2)

    axes[2, 0].set_xlabel('Video Frame Index')
    axes[2, 1].set_xlabel('Video Frame Index')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('analysis_2_error_over_time_aligned.png', dpi=300)

    # 3. 误差分布直方图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Error Distribution KDE (Aligned & Filtered)', fontsize=18, fontweight='bold')

    all_metrics = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    flat_axes = axes.flatten()

    for i, axis in enumerate(all_metrics):
        unit = 'm' if i < 3 else 'deg'
        sns.histplot(df_valid[f'err_{axis}'], ax=flat_axes[i], kde=True, bins=40, color='teal')
        mean_val = df_valid[f'err_{axis}'].mean()
        flat_axes[i].axvline(mean_val, color='red', linestyle='--', label=f"Mean Bias: {mean_val:.4f}")
        flat_axes[i].set_title(f'{axis.upper()} Error Distribution')
        flat_axes[i].set_xlabel(f'Error ({unit})')
        flat_axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('analysis_3_error_distribution_aligned.png', dpi=300)

    # 保存计算好误差的完整表格
    df_valid.to_csv('merged_error_analysis_aligned.csv', index=False)
    print("💾 所有分析完成！请查看当前目录下的 3 张图表和清洗后的汇总 CSV 文件。")


if __name__ == '__main__':
    main()