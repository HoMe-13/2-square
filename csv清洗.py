import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PoseDataCleaner:
    """
    专业的 6DoF 位姿数据清洗器 (硬阈值阻断版)。
    当前模式：基于偏航角 (Yaw) 阈值识别坏数据，并将其强制补 0。
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.cleaned_df = self.df.copy()

        # 定义需要处理的列名
        self.pos_cols = ['est_x_east(m)', 'est_y_north(m)', 'est_z_up(m)']
        self.rot_cols = ['est_roll(deg)', 'est_pitch(deg)', 'est_yaw(deg)']
        self.all_target_cols = self.pos_cols + self.rot_cols

    def detect_outliers_by_yaw_threshold(self, threshold=100.0):
        """
        基于偏航角 (Yaw) 绝对值的硬阈值检测。
        如果偏航角的绝对值大于 threshold，则判定该帧为严重的误检测。
        """
        print(f"[*] 正在执行偏航角硬阈值检测: 判定偏航角绝对值 > {threshold}° 为坏数据")

        # 使用 .abs() 获取绝对值，这样无论是 >100 还是 <-100 的极端错误都能被揪出来
        outlier_mask = self.df['est_yaw(deg)'].abs() > threshold

        return outlier_mask

    def process_and_fill_zeros(self, yaw_threshold=100.0):
        """
        执行清洗流程：识别离群点 -> 将异常位姿数据直接补 0
        """
        # 1. 仅使用最直接的 Yaw 阈值进行检测
        self.outlier_mask = self.detect_outliers_by_yaw_threshold(threshold=yaw_threshold)

        outlier_count = self.outlier_mask.sum()
        total_count = len(self.df)
        print(f"[*] 检测完成！共发现 {outlier_count} 个偏航角异常帧，占比 {(outlier_count / total_count) * 100:.2f}%")

        # 2. 拷贝原始数据，并将坏数据所在的行，在所有位姿列上赋值为 0.0
        # 完美保留原有的 Video_Frame 列，保证输出文件的帧序号绝对连续
        self.cleaned_df = self.df.copy()
        self.cleaned_df.loc[self.outlier_mask, self.all_target_cols] = 0.0

        print(f"[*] 已将这 {outlier_count} 行坏数据对应的 XYZ 和姿态角成功强制补 0。")

        return self.cleaned_df

    def calculate_and_print_stats(self):
        """
        计算并输出平均值、极差，以及两个核心平稳性指标（标准差与帧抖动）。
        (注意：统计时将严格排除那些被补 0 的坏帧，确保计算绝对纯净！)
        """
        print("\n" + "=" * 85)
        print("📊 清洗后数据的真实统计信息 (Mean, Max-Min Error, STD, Jitter) [已排除坏帧]")
        print("=" * 85)

        # 利用保存的掩码提取纯净有效数据 (Valid DataFrame)
        # ~self.outlier_mask 表示“取反”，即只保留那些正常数据参与计算
        if hasattr(self, 'outlier_mask'):
            valid_df = self.cleaned_df[~self.outlier_mask]
        else:
            valid_df = self.cleaned_df

        stats_data = []
        for col in self.all_target_cols:
            # 基础指标
            col_mean = valid_df[col].mean()
            col_max = valid_df[col].max()
            col_min = valid_df[col].min()
            col_error = col_max - col_min  # 极差 (Peak-to-Peak Error)

            # 【新增】平稳性指标 1：标准差 (Standard Deviation) - 衡量全局离散程度
            col_std = valid_df[col].std()

            # 【新增】平稳性指标 2：相邻帧抖动 (Frame-to-Frame Jitter) - 衡量局部平滑度
            # diff() 计算与上一帧的差值，abs() 取绝对值，mean() 求均值
            col_jitter = valid_df[col].diff().abs().mean()

            stats_data.append({
                'Feature': col,
                'Mean': col_mean,
                'Max-Min Error': col_error,
                'Standard Deviation': col_std,
                'Frame Jitter': col_jitter
            })

            # 精美控制台排版输出
            unit = "m" if "(m)" in col else "deg"
            name = col.split('(')[0].replace('est_', '').upper()

            # 格式化输出字符串，展示全面的评估指标
            print(
                f"[{name:^7}] 均值: {col_mean:>8.4f} {unit} | 极差: {col_error:>7.4f} | 标准差(全局平稳): {col_std:>7.4f} | 帧抖动(局部平滑): {col_jitter:>7.4f}")

        print("=" * 85 + "\n")
        return pd.DataFrame(stats_data)

    def plot_comparison(self, feature_col='est_z_up(m)'):
        """
        可视化清洗前后的对比效果
        """
        plt.figure(figsize=(14, 6))

        # 绘制原始数据
        plt.plot(self.df['Video_Frame'], self.df[feature_col],
                 label='Raw Data (With Misdetections)', color='red', alpha=0.5, linestyle='--')

        # 绘制补零后的数据
        plt.plot(self.cleaned_df['Video_Frame'], self.cleaned_df[feature_col],
                 label='Cleaned Data (Zeros Filled)', color='blue', linewidth=2)

        plt.title(f'Data Cleaning Comparison (Filled with Zeros): {feature_col}')
        plt.xlabel('Video Frame')
        plt.ylabel(feature_col)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def save(self, output_path='cleaned_zeros_222.csv'):
        """保存清洗后的数据"""
        self.cleaned_df.to_csv(output_path, index=False)
        print(f"[*] 补零后的全新数据已保存至: {output_path}")


if __name__ == "__main__":
    # 使用示例
    # 确保 222.csv 与此脚本在同一目录下
    cleaner = PoseDataCleaner(r'C:\Users\20164\Desktop\123\222.csv')

    # 1. 执行清洗：设置偏航角阈值为 100 度。一旦绝对值超过 100，整行位姿变 0。
    cleaned_data = cleaner.process_and_fill_zeros(yaw_threshold=100.0)

    # 2. 计算并在后台输出全面统计信息 (含标准差与帧抖动)
    cleaner.calculate_and_print_stats()

    # 3. 可视化清洗效果
    cleaner.plot_comparison(feature_col='est_yaw(deg)')

    # 4. 导出全新补0后的 CSV 文件
    cleaner.save('cleaned_zeros_222.csv')