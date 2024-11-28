import cv2
import numpy as np
from pathlib import Path

def segment_and_correct_painting(query_image_path, segmentation_model, confidence_threshold=0.50):
    '''
    query_image_path: path to query image file
    segmentation_model: ultralytics.YOLO(weights)
    '''
    query_img = cv2.imread(query_image_path)
    img_name = Path(query_image_path).stem
    results = segmentation_model(query_img)

    obj = results[0]

    if obj.boxes is not None and len(obj.boxes) > 0:
        confidences = obj.boxes.conf.cpu().numpy()
        high_conf_indices = np.where(confidences >= confidence_threshold)[0]

        if len(high_conf_indices) == 0:
            print(f"No objects found with confidence >= {confidence_threshold}.")
            return None

        first_high_conf_index = high_conf_indices[0]

        background_mask = np.zeros(query_img.shape[:2], dtype=np.uint8)
        contour = obj.masks.xy[first_high_conf_index].astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(background_mask, [contour], -1, (255), thickness=cv2.FILLED)

        masked_image = cv2.bitwise_and(query_img, query_img, mask=background_mask)

        contours, _ = cv2.findContours(background_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("No contours found.")
            return None

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, epsilon, True)

        if len(corners) != 4:
            print("Could not detect 4 corners of the painting.")
            return None

        corners = corners.reshape(4, 2)

        center = np.mean(corners, axis=0)

        sorted_corners = sorted(corners, key=lambda p: (np.arctan2(p[1] - center[1], p[0] - center[0])))

        top_left, top_right, bottom_right, bottom_left = sorted_corners

        src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])

        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)

        width = int(max(width_top, width_bottom))
        height = int(max(height_left, height_right))

        dst_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        corrected_image = cv2.warpPerspective(query_img, M, (width, height), flags=cv2.INTER_LINEAR)

        corrected_image_path = f"{img_name}_homography_corrected.jpg"
        cv2.imwrite(corrected_image_path, corrected_image)
        print(f"Corrected painting image saved as: {corrected_image_path}")

        corrected_image_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
        return corrected_image_rgb

    else:
        print("No objects detected.")
        return None