import cv2
import os
import glob
from pathlib import Path

# ==========================================
# ⚙️ 核心配置区 (你可以根据需要修改这些值)
# ==========================================

# 1. 你的视频存放在哪个文件夹？(支持多个视频文件)
VIDEO_INPUT_DIR = (r"C:\Users\20164\Desktop\11\square")

# 2. 提取出来的图片存放在哪个文件夹？
IMAGE_OUTPUT_DIR = r"C:\Users\20164\Desktop\11\image"

# 3. 抽帧间隔 (每隔多少帧提取一张图？)
# 假设视频是 30fps，设为 10，就是每秒提取 3 张不同姿态的图。
# 设得越小图越多，设得越大图片差异越明显。
FRAME_INTERVAL = 10

# 4. 模糊过滤阈值 (玄学调参)
# 核心原理是计算图像灰度拉普拉斯方差。方差越小说明越模糊。
# 推荐值：100~300。如果发现很多清晰的图也被扔了，就把这个值调小；如果发现保存了很多糊图，就调大。
BLUR_THRESHOLD = 50

# 5. 是否需要开启模糊过滤功能？(如果你的视频非常稳，可以关掉)
ENABLE_BLUR_FILTER = True


# ==========================================

def is_blurry(image, threshold):
    """
    基于拉普拉斯算子方差的图像模糊度检测
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算拉普拉斯方差，值越大代表图像边缘越锐利，越清晰
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance


def process_video(video_path, output_dir):
    """
    处理单个视频文件
    """
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ 错误：无法打开视频文件 {video_path}")
        return

    # 获取视频的基本信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"\n🎬 开始处理视频: {video_name} | 总帧数: {total_frames} | 帧率: {fps} FPS")

    frame_count = 0
    saved_count = 0
    blurry_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完毕

        # 1. 跳帧逻辑：只提取固定间隔的帧
        if frame_count % FRAME_INTERVAL == 0:

            save_flag = True
            blur_score = 0

            # 2. 模糊检测逻辑
            if ENABLE_BLUR_FILTER:
                is_blur, blur_score = is_blurry(frame, BLUR_THRESHOLD)
                if is_blur:
                    save_flag = False
                    blurry_count += 1

            # 3. 保存逻辑
            if save_flag:
                # 构造极具辨识度的文件名：视频名_六位帧号.jpg
                out_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
                out_path = os.path.join(output_dir, out_filename)

                cv2.imwrite(out_path, frame)
                saved_count += 1

                # 每保存 50 张打印一次进度，防止你看终端以为卡死了
                if saved_count % 50 == 0:
                    print(f"  📸 已提取 {saved_count} 张... (当前帧: {frame_count}/{total_frames})")

        frame_count += 1

    cap.release()
    print(f"✅ 视频 {video_name} 处理完成！")
    print(f"   📊 成功保存: {saved_count} 张 | 过滤模糊废片: {blurry_count} 张")


if __name__ == "__main__":
    # 1. 确保输入文件夹存在
    if not os.path.exists(VIDEO_INPUT_DIR):
        print(f"⚠️ 找不到视频文件夹 '{VIDEO_INPUT_DIR}'，请先创建并把视频放进去！")
        exit()

    # 2. 如果输出文件夹不存在，自动创建它
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

    # 3. 搜索所有的 .mp4 和 .avi 文件 (不区分大小写)
    video_files = []
    for ext in ('*.mp4', '*.avi', '*.MP4', '*.AVI','*.MOV'):
        video_files.extend(glob.glob(os.path.join(VIDEO_INPUT_DIR, ext)))

    if len(video_files) == 0:
        print(f"⚠️ 在 '{VIDEO_INPUT_DIR}' 下没有找到任何视频文件！")
        exit()

    print(f"🚀 找到 {len(video_files)} 个视频文件，准备开始疯狂抽帧...")

    # 4. 遍历处理每一个视频
    for v_path in video_files:
        process_video(v_path, IMAGE_OUTPUT_DIR)

    print(f"\n🎉 所有视频处理完毕！抽出的图片已保存在 '{IMAGE_OUTPUT_DIR}' 目录下。")
    print(f"🎯 接下来，你可以打开 LabelImg 或 MakeSense 去愉快地画框了！")