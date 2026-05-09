import cv2
import numpy as np
import pandas as pd
import os
import glob
import shutil
import matplotlib.pyplot as plt
from landmark_detector import LandmarkDetector
from pose_estimator import PoseEstimator
from yolo_detector import YOLODetector

ENABLE_YOLO_PREDETECTION = True


class BatchProcessor:
    def __init__(self, csv_file=None, images_folder=None, use_yolo=True, plot_dir=None, debug_dir=None):
        self.plot_dir = plot_dir
        self.debug_dir = debug_dir
        self.images_folder = images_folder

        self.yolo_detector = None
        if use_yolo and ENABLE_YOLO_PREDETECTION:
            try:
                self.yolo_detector = YOLODetector()
                if not self.yolo_detector.is_loaded:
                    self.yolo_detector = None
            except Exception as e:
                self.yolo_detector = None

        self.detector = LandmarkDetector(yolo_detector=self.yolo_detector)
        self.estimator = PoseEstimator()

        self.total_errors = []
        self.position_errors_2D = []
        self.rotation_errors_2D = []
        self.X_error = []
        self.Y_error = []
        self.distance = []
        self.results_data = []

        # 用于记录第一帧的绝对坐标，作为后续的相对零点
        self.origin_est_pos = None
        self.origin_est_euler = None

        if csv_file and os.path.exists(csv_file):
            self.ground_truth = pd.read_csv(csv_file)
            self.ground_truth.columns = [c.strip() for c in self.ground_truth.columns]
            self._build_image_timestamp_map()
        else:
            self.ground_truth = None

    def _build_image_timestamp_map(self):
        self.img_map = {}
        if not os.path.exists(self.images_folder): return
        files = os.listdir(self.images_folder)
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                name_without_ext = os.path.splitext(f)[0]
                try:
                    ts = float(name_without_ext)
                    self.img_map[ts] = f
                except ValueError:
                    continue

    def calculate_angle_diff(self, angle1, angle2, degrees=True):
        diff = angle1 - angle2
        if degrees:
            diff = (diff + 180) % 360 - 180
        else:
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff

    def get_mode_display_name(self, mode):
        modes = {"full": "Full (A+B+C)", "small_only": "Small (B+C)", "single_only_C": "Single (C)"}
        return modes.get(mode, str(mode))

    def draw_error_info(self, image, pos_error, rot_error, est_pos, gt_pos, est_euler, gt_euler, roi=None, mode=None):
        result_img = image.copy()
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        overlay = result_img.copy()
        cv2.rectangle(overlay, (10, 10), (650, 330), (0, 0, 0), -1)
        result_img = cv2.addWeighted(result_img, 0.7, overlay, 0.3, 0)

        y_offset = 30
        cv2.putText(result_img, "YOLO: ON" if roi else "YOLO: OFF", (20, y_offset), font, font_scale,
                    (0, 255, 0) if roi else (128, 128, 128), thickness)

        y_offset += 25
        cv2.putText(result_img, f"Mode: {self.get_mode_display_name(mode)}", (20, y_offset), font, font_scale,
                    (255, 255, 255), thickness)

        y_offset += 25
        cv2.putText(result_img, f"Pos Error 2D: {pos_error:.3f} cm", (20, y_offset), font, font_scale, (0, 255, 255),
                    thickness)
        y_offset += 25
        cv2.putText(result_img, f"Rot Error 2D (Yaw): {rot_error:.2f} deg", (20, y_offset), font, font_scale,
                    (0, 255, 255), thickness)

        y_offset += 30
        cv2.putText(result_img, f"Est Height: {abs(est_pos[2]):.3f} m", (20, y_offset), font, font_scale, (0, 255, 0),
                    thickness)

        y_offset += 30
        cv2.putText(result_img, f"Rel Est Pos: ({est_pos[0]:.2f}, {est_pos[1]:.2f}, {est_pos[2]:.2f})", (20, y_offset),
                    font, font_scale, (0, 255, 0), thickness)
        y_offset += 25
        cv2.putText(result_img, f"GT Rel Pos: ({gt_pos[0]:.2f}, {gt_pos[1]:.2f}, {gt_pos[2]:.2f})", (20, y_offset),
                    font, font_scale, (255, 255, 0), thickness)

        y_offset += 25
        cv2.putText(result_img, f"Rel Est Rot: ({est_euler[0]:.1f}, {est_euler[1]:.1f}, {est_euler[2]:.1f})",
                    (20, y_offset), font, font_scale, (0, 255, 0), thickness)
        y_offset += 25
        cv2.putText(result_img, f"GT Rel Rot: ({gt_euler[0]:.1f}, {gt_euler[1]:.1f}, {gt_euler[2]:.1f})",
                    (20, y_offset), font, font_scale, (255, 255, 0), thickness)

        if roi is not None:
            x1, y1, x2, y2 = roi
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(result_img, "YOLO ROI", (x1, y1 - 10), font, 0.5, (255, 0, 0), 2)

        return result_img

    def plot_error_analysis(self, df):
        if self.plot_dir is None or df.empty: return
        df = df.sort_values('FrameID')
        x_axis, xlabel = df['FrameID'], 'Frame ID'

        plt.figure(figsize=(10, 5))
        plt.plot(x_axis, df['PositionError_2D'], label='Pos Error 2D', color='blue', linewidth=2)
        plt.plot(x_axis, df['X_Error'], label='Err X', color='gray', linestyle='--', alpha=0.6)
        plt.plot(x_axis, df['Y_Error'], label='Err Y', color='orange', linestyle='--', alpha=0.6)
        plt.axhline(df['PositionError_2D'].mean(), color='k', linestyle='-.')
        plt.title('Position Error Analysis (2D)')
        plt.xlabel(xlabel)
        plt.ylabel('Error (cm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'position_error_analysis.png'))
        plt.close()

    def process_single_image(self, image_path, row=None):
        image = cv2.imread(image_path)
        if image is None: return None, None

        # 🌟 修复相机硬件倒装：全局翻转 180 度
        image = cv2.rotate(image, cv2.ROTATE_180)

        if self.yolo_detector is not None:
            sorted_corners, binary, squares, mode, roi = self.detector.process_image_with_yolo(image,
                                                                                               visualize_yolo=False)
        else:
            sorted_corners, binary, squares, mode = self.detector.process_image_new(image)
            roi = None

        if sorted_corners is None:
            return image, binary

        all_corners = []
        for square_corners in sorted_corners:
            all_corners.extend(square_corners)

        rvec, tvec = self.estimator.estimate_pose_pnp(all_corners, mode)
        if rvec is None or tvec is None:
            return image, binary

        square_names = {'full': ['A', 'B', 'C'], 'small_only': ['B', 'C']}.get(mode,
                                                                               [mode.split('_')[-1]] if mode else ['?'])
        result_img = self.estimator.draw_detection_results(image, sorted_corners, square_names)
        result_img = self.estimator.draw_coordinate_axes(result_img, rvec, tvec)

        rvec_mat, _ = cv2.Rodrigues(rvec)
        est_euler = self.estimator.rotation_matrix_to_euler_angles(rvec_mat)
        est_pos = tvec.ravel()

        # 🌟 相对第一帧（原点）计算误差
        if self.origin_est_pos is None:
            self.origin_est_pos = est_pos.copy()
            self.origin_est_euler = est_euler.copy()

        rel_est_pos = est_pos - self.origin_est_pos
        rel_est_pos[2] = est_pos[2]

        rel_est_euler = np.zeros(3)
        for i in range(3):
            rel_est_euler[i] = self.calculate_angle_diff(est_euler[i], self.origin_est_euler[i])

        if self.ground_truth is not None and row is not None:
            frame_id = row['FrameID']
            gt_pos = np.array([row['LandmarkPosX'], row['LandmarkPosY'], row['LandmarkPosZ']])
            gt_euler = np.array([row['roll'], row['pitch'], row['yaw']])

            pos_error_2D = np.linalg.norm(rel_est_pos[:2] - gt_pos[:2]) * 100
            rot_error_2D = abs(self.calculate_angle_diff(rel_est_euler[2], gt_euler[2]))
            X_error, Y_error = abs(rel_est_pos[0] - gt_pos[0]) * 100, abs(rel_est_pos[1] - gt_pos[1]) * 100

            self.position_errors_2D.append(pos_error_2D)
            self.rotation_errors_2D.append(rot_error_2D)
            self.X_error.append(X_error)
            self.Y_error.append(Y_error)
            self.distance.append(est_pos[2])

            self.results_data.append({
                'FrameID': frame_id, 'ImageFile': os.path.basename(image_path),
                'DetectionMode': mode, 'PositionError_2D': pos_error_2D,
                'RotationError_2D': rot_error_2D, 'X_Error': X_error, 'Y_Error': Y_error
            })

            result_img = self.draw_error_info(result_img, pos_error_2D, rot_error_2D, rel_est_pos, gt_pos,
                                              rel_est_euler, gt_euler, roi, mode)

        return result_img, binary

    def process_batch(self):
        print("\n开始批量处理并保存本地图片...")
        if not self.img_map: return

        final_dir = os.path.join(self.plot_dir, "final_result")
        os.makedirs(final_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

        for idx, row in self.ground_truth.iterrows():
            csv_ts = float(row['Timestamp'])
            closest_ts = min(self.img_map.keys(), key=lambda k: abs(k - csv_ts))

            if abs(closest_ts - csv_ts) > 0.1: continue

            image_file = self.img_map[closest_ts]
            image_path = os.path.join(self.images_folder, image_file)

            result_img, binary_img = self.process_single_image(image_path, row)

            if result_img is not None:
                # 🌟 保存最终结果图
                cv2.imwrite(os.path.join(final_dir, image_file), result_img)

                # 🌟 保存完美填充黑底的二值化图
                if binary_img is not None:
                    cv2.imwrite(os.path.join(self.debug_dir, f"bin_{image_file}"), binary_img)

                # 缩放弹窗，防止高分辨率图片撑爆屏幕
                h, w = result_img.shape[:2]
                scale = 800 / max(h, w)

                show_result = cv2.resize(result_img, (int(w * scale), int(h * scale))) if scale < 1 else result_img
                cv2.imshow("Final Result", show_result)

                if binary_img is not None:
                    show_bin = cv2.resize(binary_img, (int(w * scale), int(h * scale))) if scale < 1 else binary_img
                    cv2.imshow("Binary Map", show_bin)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if self.results_data:
            csv_path = os.path.join(self.plot_dir, "result_errors.csv")
            pd.DataFrame(self.results_data).to_csv(csv_path, index=False)
            self.plot_error_analysis(pd.DataFrame(self.results_data))
            print(f"✅ 处理完成！结果图已保存至: {final_dir}")
            print(f"✅ 二值化图已保存至: {self.debug_dir}")
            print(f"✅ 误差数据 CSV 已保存至: {csv_path}")

        cv2.destroyAllWindows()


def main():
    images_folder = r"D:\work\data\data\square0427_1\data-real"
    csv_file = r"D:\work\data\data\square0427_1\Final_Trajectory_Dataset.csv"

    vis_dir = "visualization/pose_inference"
    plot_dir = os.path.join(vis_dir, "plots")
    debug_dir = os.path.join(vis_dir, "debug_binary")

    if os.path.exists(debug_dir): shutil.rmtree(debug_dir)
    os.makedirs(debug_dir, exist_ok=True)

    processor = BatchProcessor(csv_file=csv_file, images_folder=images_folder, use_yolo=True, plot_dir=plot_dir,
                               debug_dir=debug_dir)
    processor.process_batch()


if __name__ == "__main__":
    main()