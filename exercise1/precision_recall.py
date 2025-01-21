import numpy as np
from iou import calculate_ious
from utils import get_data

def precision_recall(ious, gt_classes, pred_classes, iou_threshold=0.5):
    """
    Calculate precision and recall.
    Args:
    - ious [array]: NxM array of IoUs.
    - gt_classes [array]: 1xN array of ground truth classes.
    - pred_classes [array]: 1xM array of predicted classes.
    - iou_threshold [float]: IoU threshold to determine a match (default is 0.5).
    Returns:
    - precision [float]
    - recall [float]
    """
    # Initialize counts
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Keep track of matched ground truth boxes
    matched_gt = [False] * len(gt_classes)

    # Iterate over each predicted bounding box
    for j, pred_class in enumerate(pred_classes):
        max_iou = 0
        best_match = -1

        # Compare against all ground truth boxes
        for i, gt_class in enumerate(gt_classes):
            if matched_gt[i]:  # Skip already matched GT boxes
                continue
            
            # Check if classes match and IoU is the highest seen so far
            if pred_class == gt_class and ious[i, j] > max_iou:
                max_iou = ious[i, j]
                best_match = i

        # Determine if we have a valid match
        if max_iou >= iou_threshold and best_match != -1:
            true_positives += 1
            matched_gt[best_match] = True  # Mark the ground truth as matched
        else:
            false_positives += 1  # Unmatched prediction is a false positive

    # Count False Negatives (ground truth boxes that were never matched)
    false_negatives = matched_gt.count(False)

    # Compute Precision and Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

if __name__ == "__main__": 
    ground_truth, predictions = get_data()
    
    # Get bounding boxes and classes for a specific image
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_classes = [g['classes'] for g in ground_truth if g['filename'] == filename][0]
    
    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_classes = [p['classes'] for p in predictions if p['filename'] == filename][0]
    
    # Convert to NumPy arrays
    ious = calculate_ious(np.array(gt_bboxes), np.array(pred_bboxes))
    precision, recall = precision_recall(ious, gt_classes, pred_classes)

    print("Precision:", precision)
    print("Recall:", recall)
