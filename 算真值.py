import pandas as pd
import numpy as np
import pymap3d as pm
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


def normalize_angle(angle_deg):
    """将角度限制在 [-180, 180] 范围内"""
    return (angle_deg + 180) % 360 - 180


def generate_video_sync_ground_truth(input_file, output_file):
    extracted_data = []
    current_gps_week = None
    current_gps_seconds = None

    print(f"1. 开始读取原始数据: {input_file} ...")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            if line.startswith('<INSPVAX'):
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        current_gps_week = int(parts[5])
                        current_gps_seconds = float(parts[6])
                    except ValueError:
                        continue

            elif line.startswith('<') and 'INS_SOLUTION' in line:
                parts = line.split()
                if current_gps_week is not None and current_gps_seconds is not None and len(parts) >= 13:
                    try:
                        extracted_data.append({
                            'GPS_Seconds': current_gps_seconds,
                            'Latitude': float(parts[3]),
                            'Longitude': float(parts[4]),
                            'Height': float(parts[5]),
                            'Roll': float(parts[10]),
                            'Pitch': float(parts[11]),
                            'Azimuth': float(parts[12])
                        })
                    except ValueError:
                        pass
                    finally:
                        current_gps_week = None
                        current_gps_seconds = None

    if not extracted_data:
        print("❌ 提取失败，请检查文件格式或路径。")
        return

    df_100hz = pd.DataFrame(extracted_data)
    df_100hz = df_100hz.drop_duplicates(subset=['GPS_Seconds']).sort_values('GPS_Seconds')

    # =========================================================
    # 2. 提取第一帧作为绝对地标原点 (0, 0, 0) 与 基准姿态
    # =========================================================
    print("2. 正在提取第一帧作为绝对地标原点...")
    first_frame = df_100hz.iloc[0]

    PAD_LAT = first_frame['Latitude']
    PAD_LON = first_frame['Longitude']
    PAD_ALT = first_frame['Height']

    # INSPVAX 的方位角(Azimuth)是真北为 0 顺时针为正。
    # 转换为标准的 ENU(东北天) 右手系 Yaw (正东为 0，逆时针为正)
    PAD_YAW_ENU = normalize_angle(90.0 - first_frame['Azimuth'])

    # 构建地标的绝对旋转矩阵 R_pad (使用 ZYX 顺序，即 Yaw-Pitch-Roll)
    r_pad = R.from_euler('ZYX', [PAD_YAW_ENU, first_frame['Pitch'], first_frame['Roll']], degrees=True)

    print(f"   已锁定原点坐标 -> 纬度:{PAD_LAT:.6f}, 经度:{PAD_LON:.6f}, 高度:{PAD_ALT:.4f}")

    # =========================================================
    # 3. 计算所有帧相对于第一帧的距离和刚体姿态
    # =========================================================
    print("3. 正在进行相对 XYZ 投影与相对姿态计算...")
    # 投影为局部的 ENU (东、北、天) 笛卡尔坐标系，单位：米
    rel_east, rel_north, rel_up = pm.geodetic2enu(
        df_100hz['Latitude'].values,
        df_100hz['Longitude'].values,
        df_100hz['Height'].values,
        PAD_LAT, PAD_LON, PAD_ALT
    )
    df_100hz['rel_x_m'] = rel_east
    df_100hz['rel_y_m'] = rel_north
    df_100hz['rel_z_m'] = rel_up

    rel_rolls, rel_pitches, rel_yaws = [], [], []
    for _, row in df_100hz.iterrows():
        uav_yaw_enu = normalize_angle(90.0 - row['Azimuth'])

        # 构建无人机当前帧的绝对旋转矩阵 R_drone
        r_drone = R.from_euler('ZYX', [uav_yaw_enu, row['Pitch'], row['Roll']], degrees=True)

        # 核心：矩阵乘法求相对旋转 R_rel = (R_pad)^-1 * R_drone
        r_rel = r_pad.inv() * r_drone

        # 从相对旋转矩阵中反解出相对欧拉角
        y, p, r = r_rel.as_euler('ZYX', degrees=True)
        rel_yaws.append(y)
        rel_pitches.append(p)
        rel_rolls.append(r)

    df_100hz['rel_roll_deg'] = rel_rolls
    df_100hz['rel_pitch_deg'] = rel_pitches
    df_100hz['rel_yaw_deg'] = rel_yaws

    # =========================================================
    # 4. 30fps 重采样与精密插值
    # =========================================================
    print("4. 正在生成 30fps 视频同步时间轴并进行精密插值...")
    t_start = df_100hz['GPS_Seconds'].iloc[0]
    t_end = df_100hz['GPS_Seconds'].iloc[-1]

    fps = 30.0
    frame_interval = 1.0 / fps
    video_timestamps = np.arange(t_start, t_end, frame_interval)
    video_frames = np.arange(len(video_timestamps))

    source_times = df_100hz['GPS_Seconds'].values

    # 位置插值
    f_x = interp1d(source_times, df_100hz['rel_x_m'].values, kind='linear')
    f_y = interp1d(source_times, df_100hz['rel_y_m'].values, kind='linear')
    f_z = interp1d(source_times, df_100hz['rel_z_m'].values, kind='linear')

    # 姿态解包(Unwrap)与插值，防止 -180 到 +180 的跳变导致插值出 0
    roll_unwrapped = np.unwrap(np.deg2rad(df_100hz['rel_roll_deg'].values))
    pitch_unwrapped = np.unwrap(np.deg2rad(df_100hz['rel_pitch_deg'].values))
    yaw_unwrapped = np.unwrap(np.deg2rad(df_100hz['rel_yaw_deg'].values))

    f_roll = interp1d(source_times, roll_unwrapped, kind='linear')
    f_pitch = interp1d(source_times, pitch_unwrapped, kind='linear')
    f_yaw = interp1d(source_times, yaw_unwrapped, kind='linear')

    # 计算 30fps 目标值并转回角度
    interp_x = f_x(video_timestamps)
    interp_y = f_y(video_timestamps)
    interp_z = f_z(video_timestamps)

    interp_roll = normalize_angle(np.rad2deg(f_roll(video_timestamps)))
    interp_pitch = normalize_angle(np.rad2deg(f_pitch(video_timestamps)))
    interp_yaw = normalize_angle(np.rad2deg(f_yaw(video_timestamps)))

    # =========================================================
    # 5. 组装、格式化并保存 (彻底干掉科学计数法)
    # =========================================================
    df_video_sync = pd.DataFrame({
        'Video_Frame': video_frames,
        'Matched_GPS_Time': video_timestamps,
        'rel_x_east(m)': interp_x,
        'rel_y_north(m)': interp_y,
        'rel_z_up(m)': interp_z,
        'rel_roll(deg)': interp_roll,
        'rel_pitch(deg)': interp_pitch,
        'rel_yaw(deg)': interp_yaw
    })

    # 舍入小数位数，避免科学计数法
    df_video_sync = df_video_sync.round({
        'Matched_GPS_Time': 3,
        'rel_x_east(m)': 4,
        'rel_y_north(m)': 4,
        'rel_z_up(m)': 4,
        'rel_roll(deg)': 3,
        'rel_pitch(deg)': 3,
        'rel_yaw(deg)': 3
    })

    # 写入文件时强制使用常规浮点数格式
    df_video_sync.to_csv(output_file, index=False, float_format='%.4f')

    print("--------------------------------------------------")
    print(f"✅ 处理完成！成功生成 30fps 严谨刚体真值表。")
    print(f"📁 结果已保存至: {output_file}")
    print("\n【前 3 帧预览】 (第一帧绝对是 0.0000) :")
    print(df_video_sync.head(3).to_string(index=False))
    print("--------------------------------------------------")


if __name__ == "__main__":
    # 请在这里修改为你的实际输入和输出路径
    INPUT_FILENAME = r"C:\Users\20164\Desktop\地标真值.txt"
    OUTPUT_FILENAME = "Video_Sync_30fps_GroundTruth_Zeroed.csv"

    generate_video_sync_ground_truth(INPUT_FILENAME, OUTPUT_FILENAME)