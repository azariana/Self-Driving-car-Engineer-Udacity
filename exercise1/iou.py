import numpy as np
from utils import get_data, check_results

def calculate_iou(gt_bbox, pred_bbox):
    """
    Calculate IoU between two bounding boxes.
    Args:
    - gt_bbox [array]: 1x4 ground truth bbox
    - pred_bbox [array]: 1x4 predicted bbox
    Returns:
    - iou [float]: Intersection over Union value
    """
    # Compute intersection coordinates
    x1_inter = max(gt_bbox[0], pred_bbox[0])
    y1_inter = max(gt_bbox[1], pred_bbox[1])
    x2_inter = min(gt_bbox[2], pred_bbox[2])
    y2_inter = min(gt_bbox[3], pred_bbox[3])

    # Compute intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Compute areas of both bounding boxes
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])

    # Compute union area
    union_area = gt_area + pred_area - inter_area

    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def calculate_ious(gt_bboxes, pred_bboxes):
    """
    Calculate IoUs between multiple ground truth and predicted bounding boxes.
    Args:
    - gt_bboxes [array]: Nx4 ground truth array
    - pred_bboxes [array]: Mx4 predicted array
    Returns:
    - ious [array]: NxM array of IoUs
    """
    ious = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, pred_bbox in enumerate(pred_bboxes):
            ious[i, j] = calculate_iou(gt_bbox, pred_bbox)
    return ious

if __name__ == "__main__":
    # Load data
    ground_truth, predictions = get_data()

    # Select a specific image
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)

    # Compute IoUs
    ious = calculate_ious(gt_bboxes, pred_boxes)

    # Validate IoU results
    check_results(ious)

