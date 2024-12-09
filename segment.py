import cv2
import numpy as np
from pathlib import Path

def segment_and_correct_painting(query_image_path, segmentation_model, confidence_threshold=0.50):
    '''
    query_image_path: path to query image file
    segmentation_model: ultralytics.YOLO(weights)
    '''
    query_img = cv2.imread(query_image_path)
    results = segmentation_model(query_img, verbose=False)
    obj = results[0]

    if obj.boxes is None or len(obj.boxes) == 0:
        print("No objects found.")
        return None
    
    confidences = obj.boxes.conf.cpu().numpy()
    high_conf_indices = np.where(confidences >= confidence_threshold)[0]
    if len(high_conf_indices) == 0:
        print(f"No objects found with confidence >= {confidence_threshold}.")
        return None

    first_high_conf_index = high_conf_indices[0]
    contour = obj.masks.xy[first_high_conf_index].astype(np.int32).reshape(-1, 1, 2)

    epsilon = 0.02 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)

    if len(corners) != 4:
        print("Could not detect 4 corners of the painting.")
        return None

    corners = corners.reshape(4, 2)

    center = np.mean(corners, axis=0)
    sorted_corners = np.array(sorted(corners, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0])))

    dst_pts = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.float32([max(sorted_corners[:, 0]), max(sorted_corners[:, 1])])
    src_pts = np.float32(sorted_corners)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    width, height = int(dst_pts[1][0]), int(dst_pts[2][1])
    corrected_image = cv2.warpPerspective(query_img, M, (width, height), flags=cv2.INTER_LINEAR)

    return cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)