import numpy as np
import scipy


def eval_plane_bestmatch_normal_offset(param, gt_param, threshold=0.5):
    # Compute the L1 cost between params
    C = scipy.spatial.distance.cdist(param, gt_param, 'minkowski', p=1)

    indices = scipy.optimize.linear_sum_assignment(C)

    param_norm = np.linalg.norm(param, axis = 1) + 1e-8  # [num_pred_planes,3] -> num_pred_planes
    gt_param_norm = np.linalg.norm(gt_param, axis = 1) + 1e-8  # [num_tgt_planes, 3] -> num_tgt_planess
    param_offset = 1. / param_norm
    gt_param_offset = 1. / gt_param_norm
    param_normal = param / param_norm.reshape(-1,1)
    gt_param_normal = gt_param / gt_param_norm.reshape(-1, 1)

    angle = np.arccos(np.clip(np.dot(param_normal, gt_param_normal.T), -1.0, 1.0))  # [num_pred_planes, num_tgt_planes]
    degree = np.degrees(angle)
    bestmatch_degree = degree[indices]  # [min(num_pred_planes,num_tgt_planes),]

    bestmatch_diff = np.abs(param_offset[indices[0]] - gt_param_offset[indices[1]])

    normal_error = np.mean(bestmatch_degree)
    offset_error = np.mean(bestmatch_diff)

    return normal_error, offset_error
