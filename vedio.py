import cv2
import os
import glob


def images_to_video(image_folder, output_video_path, fps=30):
    """
    将文件夹中的序列图片合并为视频
    :param image_folder: 存放图片的文件夹路径
    :param output_video_path: 输出视频的完整路径 (如 'output.mp4')
    :param fps: 视频的播放帧率 (默认30)
    """
    print(f"📂 正在读取文件夹: {image_folder}")

    # 获取所有 jpg/png 图片，并按文件名排序
    # 这里假设你的文件名是补零的，如 frame_000000.jpg，所以直接 sort() 即可完美排序
    search_path = os.path.join(image_folder, "*.*")
    images = [img for img in glob.glob(search_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
    images.sort()

    if not images:
        print("❌ 错误：在指定的文件夹中没有找到任何图片！")
        return

    # 读取第一张图片来获取视频的宽度和高度
    first_image_path = images[0]
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"❌ 错误：无法读取第一张图片 {first_image_path}")
        return

    height, width, layers = frame.shape
    size = (width, height)
    print(f"📐 视频分辨率识别为: {width} x {height}")
    print(f"🎞️ 准备合成视频，目标帧率: {fps} FPS，总帧数: {len(images)}")

    # 初始化 OpenCV 的 VideoWriter
    # 'mp4v' 是 MP4 格式常用的编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    # 遍历所有图片并写入视频
    for i, image_path in enumerate(images):
        frame = cv2.imread(image_path)

        # 防止某张图片损坏导致崩溃
        if frame is None:
            print(f"⚠️ 警告：跳过损坏的图片 {image_path}")
            continue

        video_writer.write(frame)

        # 打印进度条
        if (i + 1) % 50 == 0 or i == len(images) - 1:
            progress = (i + 1) / len(images) * 100
            print(f"⏳ 处理进度: {i + 1}/{len(images)} ({progress:.1f}%)")

    # 释放资源
    video_writer.release()
    print(f"\n🎉 视频合成完毕！已保存至: {output_video_path}")


if __name__ == "__main__":
    # ================= 配置区 =================

    # 1. 你存放图片的文件夹路径（注意改成你电脑上的实际路径）
    INPUT_IMAGE_FOLDER = r"D:\KETI_UAV\data\data_truth\StereoData_20260420_202515\cam0_left"

    # 2. 输出视频的保存路径和文件名
    OUTPUT_VIDEO_NAME = r"C:\Users\20164\Desktop\11\甲板\vertical.mp4"

    # 3. 视频播放帧率 (设为24或30都可以，不影响RK3588测算力极限)
    VIDEO_FPS = 30

    # ==========================================

    images_to_video(INPUT_IMAGE_FOLDER, OUTPUT_VIDEO_NAME, VIDEO_FPS)