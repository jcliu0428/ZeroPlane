import numpy as np
import cv2

# https://github.com/IceTTTb/PlaneTR3D/
# Jiachen note: in original code, it first resize both pred and gt to (640, 480)
# but we only keep original inference and test (256, 192) depth
def evaluateDepths(predDepths, gtDepths, pred_mask, gt_mask, file_name=None, printInfo=False, max_depth=100):
    """Evaluate depth reconstruction accuracy"""
    predDepths = predDepths.copy()
    gtDepths = gtDepths.copy()

    pred_mask = pred_mask.copy() < 20

    # gt_mask = cv2.resize((gt_mask < 20).astype(np.uint8), (640, 480)) > 0
    # pred_mask = cv2.resize((pred_mask < 20).astype(np.uint8), (640, 480)) > 0

    # gtDepths = cv2.resize(gtDepths, (640, 480))
    # predDepths = cv2.resize(predDepths, (640, 480))

    valid_gt_mask = (gtDepths > 1e-4) * (gtDepths < max_depth)

    if gt_mask is not None:
        gt_mask = gt_mask.copy() < 20
        valid_eval_mask = valid_gt_mask * gt_mask * pred_mask

    else:
        valid_eval_mask = valid_gt_mask * pred_mask

    gtDepths = gtDepths[valid_eval_mask]

    # constrain the max pred depth
    predDepths = predDepths[valid_eval_mask]
    predDepths[predDepths > max_depth] = max_depth

    masks = gtDepths > 1e-4
    numPixels = float(masks.sum())

    rmse = np.sqrt((pow(predDepths - gtDepths, 2) * masks).sum() / numPixels)
    rmse_log = np.sqrt(
        (pow(np.log(np.maximum(predDepths, 1e-4)) - np.log(np.maximum(gtDepths, 1e-4)), 2) * masks).sum() / numPixels)
    log10 = (np.abs(
        np.log10(np.maximum(predDepths, 1e-4)) - np.log10(np.maximum(gtDepths, 1e-4))) * masks).sum() / numPixels
    rel = (np.abs(predDepths - gtDepths) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    rel_sqr = (pow(predDepths - gtDepths, 2) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    deltas = np.maximum(predDepths / np.maximum(gtDepths, 1e-4), gtDepths / np.maximum(predDepths, 1e-4)) + (
                1 - masks.astype(np.float32)) * 10000

    accuracy_1 = (deltas < 1.25).sum() / numPixels
    accuracy_2 = (deltas < pow(1.25, 2)).sum() / numPixels
    accuracy_3 = (deltas < pow(1.25, 3)).sum() / numPixels

    if printInfo:
        print(('depth statistics', rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3))
        pass

    return np.array([rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3])
