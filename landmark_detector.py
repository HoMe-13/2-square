import cv2
import numpy as np
from config import *


class LandmarkDetector:
    def __init__(self, yolo_detector=None):
        self.yolo_detector = yolo_detector
        self.subpixel_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)

    def detect_squares(self, image, offset=(0, 0)):
        h_img, w_img = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        debug_bin = np.zeros((h_img, w_img), dtype=np.uint8)

        # 🌟 步骤 1：还原原版的超强绿色抑制！
        b, g, r = cv2.split(image)
        b_f, g_f, r_f = b.astype(np.float32), g.astype(np.float32), r.astype(np.float32)
        feature_map = (r_f + b_f) - 3* g_f
        feature_map = np.clip(feature_map, 0, 255).astype(np.uint8)

        found_A, found_B, found_C = None, None, None

        # 🌟 寻找大黑框 A
        smooth_a = cv2.bilateralFilter(feature_map, 9, 75, 75)
        _, bin_a = cv2.threshold(smooth_a, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bin_a = cv2.morphologyEx(bin_a, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        debug_bin = cv2.bitwise_or(debug_bin, bin_a)

        # 只要外轮廓，保证框不乱飘
        cnts_a, _ = cv2.findContours(bin_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(cnts_a, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            if area < 5000: continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                margin = 3
                if any(p[0][0] <= margin or p[0][0] >= w_img - margin or p[0][1] <= margin or p[0][1] >= h_img - margin
                       for p in approx):
                    continue
                found_A = approx
                break

        # 🌟 步骤 2：寻找内部白框 B 和 C
        search_mask = np.zeros_like(gray)
        if found_A is not None:
            cv2.drawContours(search_mask, [found_A], -1, 255, -1)
        else:
            search_mask.fill(255)

        _, max_val, _, _ = cv2.minMaxLoc(gray, mask=search_mask)
        thresh_val = max_val * 0.80
        _, bin_bc = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        bin_bc = cv2.bitwise_and(bin_bc, bin_bc, mask=search_mask)

        kernel_bc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_bc = cv2.morphologyEx(bin_bc, cv2.MORPH_OPEN, kernel_bc)

        debug_bin = cv2.bitwise_or(debug_bin, bin_bc)

        cnts_bc, _ = cv2.findContours(bin_bc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bc_list = []
        for cnt in sorted(cnts_bc, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            if area < 100: continue
            if found_A is not None and area > cv2.contourArea(found_A) * 0.8: continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                bc_list.append(approx)

        if len(bc_list) >= 2:
            found_B = bc_list[0]
            found_C = bc_list[1]
        elif len(bc_list) == 1:
            found_C = bc_list[0]

        # 🌟 步骤 3：模式判定
        final_list = []
        mode = "none"

        if found_A is not None and found_B is not None and found_C is not None:
            final_list, mode = [found_A, found_B, found_C], "full"
        elif found_A is None and found_B is not None and found_C is not None:
            final_list, mode = [found_B, found_C], "small_only"
        elif found_C is not None and found_A is None and found_B is None:
            final_list, mode = [found_C], "single_only_C"
        else:
            mode = "none"

        if mode == "none":
            return None, debug_bin, "none"

        res_sqs = [s.reshape(4, 2).astype(np.float32) for s in final_list]
        return res_sqs, debug_bin, mode

    def refine_corners(self, gray, squares):
        refined = []
        for sq in squares:
            p = sq.reshape(-1, 1, 2).astype(np.float32)
            rp = cv2.cornerSubPix(gray, p, (7, 7), (-1, -1), self.subpixel_criteria)
            refined.append(rp.reshape(4, 2))
        return refined

    def _sort_corners_by_topology(self, pts, local_center, vec_c_to_b):
        """
        🌟 基于目标 C->B 向量的绝对角度排序法 (完美匹配 config.py)
        """
        # 1. 计算从 C 指向 B 的向量在图像上的绝对角度 (此时 vec_c_to_b 指向物理的 -Y 轴，也就是视觉上的“上方”)
        dy = vec_c_to_b[1]
        dx = vec_c_to_b[0]
        angle_up = np.degrees(np.arctan2(dy, dx))

        # 2. 根据推导，计算四个角点相对于中心的标准角度
        # 在 OpenCV 坐标系中，顺时针角度为正，逆时针为负。
        # 如果 angle_up 是“上”，那么“右”就是 angle_up + 90
        # 右上角 = angle_up + 45
        # 右下角 = angle_up + 135
        # 左下角 = angle_up + 225 (或者 -135)
        # 左上角 = angle_up - 45

        # 严格对应 config.py 的顺序：
        ideal_angles = [
            angle_up + 45,  # 0: [ 1, -1] -> 右上角
            angle_up + 135,  # 1: [ 1,  1] -> 右下角
            angle_up - 135,  # 2: [-1,  1] -> 左下角
            angle_up - 45  # 3: [-1, -1] -> 左上角
        ]

        sorted_pts = []
        available_pts = list(pts)

        for target_angle in ideal_angles:
            best_pt = None
            min_diff = 999
            for pt in available_pts:
                vec = pt - local_center
                # 计算当前角点的实际角度
                angle = np.degrees(np.arctan2(vec[1], vec[0]))

                # 计算角度差 (处理 360 度闭环)
                diff = abs((angle - target_angle + 180) % 360 - 180)
                if diff < min_diff:
                    min_diff = diff
                    best_pt = pt

            sorted_pts.append(best_pt)
            if best_pt is not None:
                available_pts = [p for p in available_pts if not np.array_equal(p, best_pt)]

        return np.array(sorted_pts, dtype=np.float32)
    def process_image_new(self, image, offset=(0, 0)):
        res_sqs, binary, mode = self.detect_squares(image, offset)
        if mode == "none":
            return None, binary, [], "none"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        refined_list = self.refine_corners(gray, res_sqs)

        # 🌟 核心：计算物理世界的方向参照物 (C -> B 向量)
        vec_c_to_b = np.array([0, -1], dtype=np.float32)

        if mode == "full":
            A_corners, B_corners, C_corners = refined_list
            center_A = np.mean(A_corners, axis=0)
            center_1 = np.mean(B_corners, axis=0)
            center_2 = np.mean(C_corners, axis=0)

            # 距离大黑框A中心远的是B，近的是C
            if np.linalg.norm(center_1 - center_A) > np.linalg.norm(center_2 - center_A):
                center_B, center_C = center_1, center_2
            else:
                center_B, center_C = center_2, center_1
            vec_c_to_b = center_B - center_C

        elif mode == "small_only":
            # 面积大的是B[0]，小的是C[1]
            center_B = np.mean(refined_list[0], axis=0)
            center_C = np.mean(refined_list[1], axis=0)
            vec_c_to_b = center_B - center_C

        final_sorted = []
        for sq in refined_list:
            final_sorted.append(self._sort_corners_by_topology(sq, np.mean(sq, axis=0), vec_c_to_b))

        return final_sorted, binary, refined_list, mode

    def process_image_with_yolo(self, image, visualize_yolo=True):
        if self.yolo_detector is None:
            res = self.process_image_new(image)
            return res[0], res[1], res[2], res[3], None

        detections, roi = self.yolo_detector.detect(image, visualize=visualize_yolo)
        if roi is None:
            res = self.process_image_new(image)
            return res[0], res[1], res[2], res[3], None

        roi_img, offset = self.yolo_detector.extract_roi(image, roi)
        res = self.process_image_new(roi_img, offset=offset)

        if res[0] is None:
            return None, res[1], res[2], res[3], roi

        final_corners = self.yolo_detector.transform_corners_to_original(res[0], offset)
        return final_corners, res[1], res[2], res[3], roi