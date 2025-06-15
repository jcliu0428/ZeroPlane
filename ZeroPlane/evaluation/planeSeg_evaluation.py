# https://github.com/facebookresearch/Mask2Former
#

import itertools
import json
import logging
import numpy as np
import os
import os.path as osp
from os.path import join as pjoin
from collections import OrderedDict
from tqdm import tqdm
import torch

from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False

from ..utils.disp import (
    visualizationBatch,
    plot_depth_recall_curve,
    plot_normal_recall_curve,
    labelcolormap
)
from ..utils.metrics import (
    evaluateMasks,
    eval_plane_recall_depth,
    eval_plane_recall_normal,
)
from ..utils.metrics_de import evaluateDepths

from ..utils.metrics_onlyparams import eval_plane_bestmatch_normal_offset


class PlaneSegEvaluator(DatasetEvaluator):
    """
    Evaluate plane segmentation metrics.
    """
    eval_iter = 0

    def __init__(
        self,
        dataset_name,
        output_dir=None,
        *,
        num_planes=None,
        vis = False,
        vis_period = 10,
        eval_period = 500,
        infer_only = False,
        save_ply = True,
        large_resolution_eval=False
    ):
        self._logger = logging.getLogger(__name__)
        if num_planes is not None:
            self._logger.warn(
                "PlaneSegEvaluator(num_planes) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name.split('_')[1] + '_' + dataset_name.split('_')[2]
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._num_planes = num_planes
        self._num_queries = num_planes + 1 if "npr" in dataset_name else num_planes  # TODO: add npr
        self.vis = vis

        # NOTE: this visualization are not order-preserved
        self.vis_period = vis_period
        self.eval_period = eval_period

        self.dataset_name = dataset_name

        if 'scannet' in dataset_name or 'nyu' in dataset_name or 'diode' in dataset_name or 'sevenscenes' in dataset_name:
            self.max_depth = 10

        elif 'mp3d' in dataset_name or 'replica' in dataset_name or 'hm3d' in dataset_name or 'taskonomy' in dataset_name or 'indoor_mixed' in dataset_name:
            self.max_depth = 30

        elif 'outdoor_mixed' in dataset_name or 'mixed' in dataset_name or 'apollo_stereo' in dataset_name \
            or 'syn' in dataset_name or 'vkitti' in dataset_name or 'parallel_domain' in dataset_name \
            or 'sanpo_synthetic' in dataset_name:
            self.max_depth = 100

        # wild data
        else:
            self.max_depth = -1

        self.test_data_num = 0

        self.eval_scale_aligned_depth = False
        self.infer_only = infer_only

        self.save_ply = save_ply

        self.large_resolution_eval = large_resolution_eval

    def reset(self):

        self.RI_VI_SC = []
        self.pixelDepth_recall_curve_of_GTpd = np.zeros((13))
        self.planeDepth_recall_curve_of_GTpd = np.zeros((13, 3))

        self.plane_frompixel_Depth_recall_curve_of_GTpd = np.zeros((13, 3))

        self.pixelNorm_recall_curve = np.zeros((13))
        self.planeNorm_recall_curve = np.zeros((13, 3))

        self.bestmatch_normal_errors = []

        self.depth_estimation_metrics = np.zeros((8))  # rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3
        self.plane_depth_from_pixel_estimation_metrics = np.zeros((8))
        self.pixel_depth_estimation_metrics = np.zeros((8))

        if self.vis:
            self.vis_dicts = []
            self.gt_vis_dicts = []
            self.file_names = []

            self.max_gt_depths = []

    def align_depth_scale(self, depth, gt, mask=None, max_depth=10):
        if mask is None:
            mask = (depth > 1e-3) * (depth < max_depth) * (gt > 1e-3) * (gt < max_depth)

        scale = np.median(gt[mask]) / (np.median(depth[mask]) + 1e-4)
        a_depth = depth * scale

        return a_depth, scale

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """

        for input, output in zip(inputs, outputs):
            self.test_data_num += 1

            sem_seg = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(sem_seg).astype(int)
            plane_depth = output["planes_depth"].to(self._cpu_device).numpy()
            seg_depth = output["seg_depth"].to(self._cpu_device).numpy()
            valid_params = output["valid_params"].to(self._cpu_device)

            k_inv_dot_xy1 = output['K_inv_dot_xy_1'].cpu().numpy()

            pred_global_pixel_normal = output['global_pixel_normal']

            gt_filename = input["npz_file_name"]
            npz_data = np.load(gt_filename)

            if 'raw_depth' in npz_data.files:
                if self.large_resolution_eval:
                    gt_raw_depth = npz_data['high_res_raw_depth']

                else:
                    gt_raw_depth = npz_data["raw_depth"]

                if 'segmentation' in npz_data.files:
                    gt = npz_data["segmentation"]
                    gt_params = npz_data["plane"]

                    if self.large_resolution_eval:
                        gt_plane_depth = npz_data["high_res_depth"][0] # b#??, h, w

                    else:
                        gt_plane_depth = npz_data['depth'][0]

                    if self.large_resolution_eval:
                        gt = cv2.resize(gt, (640, 480), interpolation=cv2.INTER_NEAREST)
                        gt_raw_depth = cv2.resize(gt_raw_depth, (640, 480), interpolation=cv2.INTER_NEAREST)
                        gt_plane_depth = cv2.resize(gt_plane_depth, (640, 480), interpolation=cv2.INTER_NEAREST)

                    gt_global_pixel_normal = input['gt_global_pixel_normal']

            if self.vis:
                if self.large_resolution_eval:
                    image = npz_data['raw_image']

                else:
                    image = npz_data["image"]  #BGR

                file_name = os.path.split(input["npz_file_name"])[-1].split(".")[0]

                if 'segmentation' in npz_data.files:
                    # gt_raw_depth = npz_data['raw_depth']
                    self.max_gt_depths.append(np.percentile(gt_raw_depth, 90))

                self.vis_dicts.append({
                    'image': image,
                    'segmentation': pred,
                    'depth_predplane': plane_depth,
                    'K_inv_dot_xy_1': k_inv_dot_xy1,
                    'pixel_normal': pred_global_pixel_normal
                })

                if seg_depth is not None:
                    self.vis_dicts[-1]['pixel_depth'] = seg_depth

                self.file_names.append(file_name)

                if 'segmentation' in npz_data.files:
                    self.gt_vis_dicts.append({
                        'image': image,
                        'segmentation': gt,
                        'depth_GTplane': gt_plane_depth,
                        'K_inv_dot_xy_1': k_inv_dot_xy1,
                        'pixel_normal': gt_global_pixel_normal,
                        'pixel_depth': gt_raw_depth
                        })

            if not self.infer_only:
                if 'scannet' in gt_filename or 'mp3d' in gt_filename or 'nyu' in gt_filename or 'replica' in gt_filename or 'hm3d' in gt_filename or 'diode' in gt_filename or 'taskonomy' in gt_filename or 'sevenscenes' in gt_filename:
                    self.eval_indoor = True

                else:
                    self.eval_indoor = False

                if self.eval_scale_aligned_depth:
                    plane_depth, _ = self.align_depth_scale(plane_depth, gt_raw_depth, max_depth=self.max_depth)

                self.RI_VI_SC.append(evaluateMasks(pred, gt, device = "cuda",  pred_non_plane_idx = self._num_planes+1, gt_non_plane_idx=self._num_planes, printInfo=False))

                # ----------------------------------------------------- evaluation
                # 1 evaluation: plane segmentation
                valid_plane_num = len(valid_params)
                pixelStatistics, planeStatistics = eval_plane_recall_depth(
                    pred, gt, plane_depth, gt_plane_depth, valid_plane_num, eval_indoor=self.eval_indoor)
                self.pixelDepth_recall_curve_of_GTpd += np.array(pixelStatistics)
                self.planeDepth_recall_curve_of_GTpd += np.array(planeStatistics)

                # _, plane_frompixel_Statistics = eval_plane_recall_depth(
                #     pred, gt, plane_from_pixel_depth, gt_plane_depth, valid_plane_num, eval_indoor=self.eval_indoor
                # )
                # self.plane_frompixel_Depth_recall_curve_of_GTpd += np.array(plane_frompixel_Statistics)

                # 2 evaluation: plane segmentation
                instance_param = valid_params.cpu().numpy()
                plane_recall, pixel_recall = eval_plane_recall_normal(pred, gt,
                                                                      instance_param, gt_params)
                self.pixelNorm_recall_curve += pixel_recall
                self.planeNorm_recall_curve += plane_recall

                instance_param = valid_params.numpy()

                try:
                    normal_error, _ = eval_plane_bestmatch_normal_offset(instance_param, gt_params)
                    self.bestmatch_normal_errors.append(normal_error)

                except:
                    print('warning: the normal error and offset error contain nan, skip...')

                # if "nyuv2_plane" in self._dataset_name or "apollo" in self._dataset_name:
                self.depth_estimation_metrics += evaluateDepths(plane_depth, gt_raw_depth, pred, gt, None, False, max_depth=self.max_depth)
                # self.plane_depth_from_pixel_estimation_metrics += evaluateDepths(plane_from_pixel_depth, gt_raw_depth, pred, gt, None, False, max_depth=self.max_depth)
                self.pixel_depth_estimation_metrics += evaluateDepths(seg_depth, gt_raw_depth, pred, gt, None, False, max_depth=self.max_depth)

        if self._output_dir:
            vis_path = pjoin(self._output_dir, "vis_" + str(PlaneSegEvaluator.eval_iter))
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)

            if self.vis:
                if (self.test_data_num - 1) % self.vis_period == 0:

                    if len(self.max_gt_depths) > 0:
                        gt_max_depth = self.max_gt_depths[-1]

                    else:
                        gt_max_depth = None

                    if len(self.gt_vis_dicts) > 0:
                        visualizationBatch(root_path = vis_path, idx = self.file_names[-1], info = "gt",
                                           data_dict = self.gt_vis_dicts[-1], num_queries = self._num_queries, save_image = True, save_segmentation = True,
                                           save_depth = True, save_ply = self.save_ply, save_cloud = False, gt_max_depth=gt_max_depth)

                    visualizationBatch(root_path = vis_path, idx = self.file_names[-1], info = "pred",
                                        data_dict = self.vis_dicts[-1], num_queries = self._num_queries, save_image = True, save_segmentation = True,
                                        save_depth = True, save_ply = self.save_ply, save_cloud = False, gt_max_depth=gt_max_depth)

                    stack_vis = True

                    if stack_vis:
                        img = cv2.imread(osp.join(vis_path, str(self.file_names[-1]) + '_image.png'))
                        seg_pred = cv2.imread(osp.join(vis_path, str(self.file_names[-1]) + '_seg_pred_blend.png'))
                        depth_pred = cv2.imread(osp.join(vis_path, str(self.file_names[-1]) + '_depth_predplane_pred.png'))

                        pred_vis = np.hstack([img, seg_pred, depth_pred])
                        pred_pixel_normal = self.vis_dicts[-1]['pixel_normal']

                        if pred_pixel_normal is not None:
                            pred_normal_vis, _ = self.vis_normal(pred_pixel_normal, None)
                            pred_vis = np.hstack([pred_vis, pred_normal_vis])

                        pred_planar_mask = self.vis_dicts[-1]['segmentation'] != 20

                        pred_pixel_depth = cv2.imread(osp.join(vis_path, str(self.file_names[-1]) + '_pixel_depth_pred.png'))
                        if pred_pixel_depth is not None:
                            pred_vis = np.hstack([pred_vis, pred_pixel_depth])

                        if len(self.gt_vis_dicts) > 0:
                            img = cv2.imread(osp.join(vis_path, str(self.file_names[-1]) + '_image.png'))
                            seg_gt = cv2.imread(osp.join(vis_path, str(self.file_names[-1]) + '_seg_gt_blend.png'))
                            depth_gt = cv2.imread(osp.join(vis_path, str(self.file_names[-1]) + '_depth_GTplane_gt.png'))

                            gt_vis = np.hstack([img, seg_gt, depth_gt])

                            if pred_pixel_normal is not None:
                                gt_pixel_normal = self.gt_vis_dicts[-1]['pixel_normal']

                                _, gt_normal_vis = self.vis_normal(pred_pixel_normal, gt_pixel_normal)
                                gt_vis = np.hstack([gt_vis, gt_normal_vis])

                            if pred_pixel_depth is not None:
                                gt_pixel_depth = cv2.imread(osp.join(vis_path, str(self.file_names[-1]) + '_pixel_depth_gt.png'))
                                gt_vis = np.hstack([gt_vis, gt_pixel_depth])

                            if pred_vis.shape == gt_vis.shape:
                                vis_all = np.vstack([pred_vis, (np.ones((10, gt_vis.shape[1], 3)) * 255).astype(np.uint8), gt_vis])

                            else:
                                vis_all = pred_vis

                        else:
                            vis_all = pred_vis

                        cv2.imwrite(osp.join(vis_path, str(self.file_names[-1]) + '_all.png'), vis_all)

    def vis_normal(self, pred_normal_map, gt_normal_map):
        if pred_normal_map is not None:
            pred_normal_map = pred_normal_map.clamp(min=-1, max=1)
            pred_normal_map  = (pred_normal_map + 1) / 2.0

            pred_normal_vis = (pred_normal_map.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred_normal_vis = pred_normal_vis[..., ::-1]

        else:
            pred_normal_vis = np.zeros((192, 256, 3)).astype(np.uint8)

        if gt_normal_map is None:
            gt_normal_vis = np.zeros((192, 256, 3)).astype(np.uint8)

        else:
            gt_normal_map = gt_normal_map.clamp(min=-1, max=1)
            gt_normal_map = (gt_normal_map + 1) / 2.0

            invalid_mask = torch.norm(gt_normal_map, dim=0) < 1e-4
            gt_normal_vis = (gt_normal_map.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_normal_vis[invalid_mask.cpu().numpy()] = 0

            gt_normal_vis = gt_normal_vis[..., ::-1]

        return pred_normal_vis, gt_normal_vis

    def evaluate(self):

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        if not self.infer_only:
            PlaneSegEvaluator.eval_iter += self.eval_period

            res = {}

            if not 'raw_kitti' in self.dataset_name:
                res_RI_VI_SC = np.sum(self.RI_VI_SC, axis=0) / len(self.RI_VI_SC)
                res["RI"] = res_RI_VI_SC[0]
                res["VI"] = res_RI_VI_SC[1]
                res["SC"] = res_RI_VI_SC[2]

                # if "nyuv2_plane" in self._dataset_name or 'apollo' in self._dataset_name:
                # rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3
                res_depth_estimation_metrics = self.depth_estimation_metrics / len(self.RI_VI_SC)
                res["DE_rel"], res["DE_rel_sqr"], res["DE_log10"], res["DE_rmse"], \
                res["DE_rmse_log"], res["DE_accuracy_1"], res["DE_accuracy_2"], res["DE_accuracy_3"] = res_depth_estimation_metrics

                res_pixel_depth_estimation_metrics = self.pixel_depth_estimation_metrics / len(self.RI_VI_SC)
                res["pixel_DE_rel"], res["pixel_DE_rel_sqr"], res["pixel_DE_log10"], res["pixel_DE_rmse"], \
                res["pixel_DE_rmse_log"], res["pixel_DE_accuracy_1"], res["pixel_DE_accuracy_2"], res["pixel_DE_accuracy_3"] = res_pixel_depth_estimation_metrics

                # res_plane_depth_from_pixel_estimation_metrics = self.plane_depth_from_pixel_estimation_metrics / len(self.RI_VI_SC)
                # res["plane_from_pixel_DE_rel"], res["plane_from_pixel_DE_rel_sqr"], res["plane_from_pixel_DE_log10"], res["plane_from_pixel_DE_rmse"], \
                # res["plane_from_pixel_DE_rmse_log"], res["plane_from_pixel_DE_accuracy_1"], res["plane_from_pixel_DE_accuracy_2"], res["plane_from_pixel_DE_accuracy_3"] = res_plane_depth_from_pixel_estimation_metrics

            else:
                res_depth_estimation_metrics = self.depth_estimation_metrics / self.test_data_num
                res["DE_rel"], res["DE_rel_sqr"], res["DE_log10"], res["DE_rmse"], \
                res["DE_rmse_log"], res["DE_accuracy_1"], res["DE_accuracy_2"], res["DE_accuracy_3"] = res_depth_estimation_metrics

                res_pixel_depth_estimation_metrics = self.pixel_depth_estimation_metrics / self.test_data_num
                res["pixel_DE_rel"], res["pixel_DE_rel_sqr"], res["pixel_DE_log10"], res["pixel_DE_rmse"], \
                res["pixel_DE_rmse_log"], res["pixel_DE_accuracy_1"], res["pixel_DE_accuracy_2"], res["pixel_DE_accuracy_3"] = res_pixel_depth_estimation_metrics

                # res_plane_depth_from_pixel_estimation_metrics = self.plane_depth_from_pixel_estimation_metrics / self.test_data_num
                # res["plane_from_pixel_DE_rel"], res["plane_from_pixel_DE_rel_sqr"], res["plane_from_pixel_DE_log10"], res["plane_from_pixel_DE_rmse"], \
                # res["plane_from_pixel_DE_rmse_log"], res["plane_from_pixel_DE_accuracy_1"], res["plane_from_pixel_DE_accuracy_2"], res["plane_from_pixel_DE_accuracy_3"] = res_plane_depth_from_pixel_estimation_metrics

        if self._output_dir:

            vis_path = pjoin(self._output_dir, "vis_" + str(PlaneSegEvaluator.eval_iter))
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)

            if self.vis and False:

                for i in tqdm(range(len(self.vis_dicts)), desc='saving visualization...'):

                    if i % self.vis_period == 0:

                        if len(self.max_gt_depths) > 0:
                            gt_max_depth = self.max_gt_depths[i]

                        else:
                            gt_max_depth = None

                        if len(self.gt_vis_dicts) > 0:
                            visualizationBatch(root_path = vis_path, idx = self.file_names[i], info = "gt",
                                               data_dict = self.gt_vis_dicts[i], num_queries = self._num_queries, save_image = True, save_segmentation = True,
                                               save_depth = True, save_ply = self.save_ply, save_cloud = False, gt_max_depth=gt_max_depth)

                        visualizationBatch(root_path = vis_path, idx = self.file_names[i], info = "pred",
                                           data_dict = self.vis_dicts[i], num_queries = self._num_queries, save_image = True, save_segmentation = True,
                                           save_depth = True, save_ply = self.save_ply, save_cloud = False, gt_max_depth=gt_max_depth)

                        stack_vis = True

                        if stack_vis:
                            img = cv2.imread(osp.join(vis_path, str(self.file_names[i]) + '_image.png'))
                            seg_pred = cv2.imread(osp.join(vis_path, str(self.file_names[i]) + '_seg_pred_blend.png'))
                            depth_pred = cv2.imread(osp.join(vis_path, str(self.file_names[i]) + '_depth_predplane_pred.png'))

                            pred_vis = np.hstack([img, seg_pred, depth_pred])
                            pred_pixel_normal = self.vis_dicts[i]['pixel_normal']

                            if pred_pixel_normal is not None:
                                pred_normal_vis, _ = self.vis_normal(pred_pixel_normal, None)
                                pred_vis = np.hstack([pred_vis, pred_normal_vis])

                            pred_planar_mask = self.vis_dicts[i]['segmentation'] != 20

                            pred_pixel_depth = cv2.imread(osp.join(vis_path, str(self.file_names[i]) + '_pixel_depth_pred.png'))
                            if pred_pixel_depth is not None:
                                pred_vis = np.hstack([pred_vis, pred_pixel_depth])

                            if len(self.gt_vis_dicts) > 0:
                            # if not self.infer_only:
                                if not 'raw_kitti' in self.dataset_name:
                                    img = cv2.imread(osp.join(vis_path, str(self.file_names[i]) + '_image.png'))
                                    seg_gt = cv2.imread(osp.join(vis_path, str(self.file_names[i]) + '_seg_gt_blend.png'))
                                    depth_gt = cv2.imread(osp.join(vis_path, str(self.file_names[i]) + '_depth_GTplane_gt.png'))

                                    gt_vis = np.hstack([img, seg_gt, depth_gt])

                                    if pred_pixel_normal is not None:
                                        gt_pixel_normal = self.gt_vis_dicts[i]['pixel_normal']

                                        _, gt_normal_vis = self.vis_normal(pred_pixel_normal, gt_pixel_normal)
                                        gt_vis = np.hstack([gt_vis, gt_normal_vis])

                                    if pred_pixel_depth is not None:
                                        gt_pixel_depth = cv2.imread(osp.join(vis_path, str(self.file_names[i]) + '_pixel_depth_gt.png'))
                                        gt_vis = np.hstack([gt_vis, gt_pixel_depth])

                                    if pred_vis.shape == gt_vis.shape:
                                        vis_all = np.vstack([pred_vis, (np.ones((10, gt_vis.shape[1], 3)) * 255).astype(np.uint8), gt_vis])

                                    else:
                                        vis_all = pred_vis

                                else:
                                    vis_all = pred_vis

                            else:
                                vis_all = pred_vis

                            cv2.imwrite(osp.join(vis_path, str(self.file_names[i]) + '_all.png'), vis_all)

            if not self.infer_only:
                if not 'raw_kitti' in self.dataset_name:
                    recall_curve_save_path = pjoin(vis_path, "recall_curve")
                    if not os.path.exists(recall_curve_save_path):
                        os.makedirs(recall_curve_save_path)

                    mine_recalls_pixel = {"zeroplane": self.pixelDepth_recall_curve_of_GTpd / len(self.RI_VI_SC) * 100}
                    mine_recalls_plane = {"zeroplane": self.planeDepth_recall_curve_of_GTpd[:, 0] / self.planeDepth_recall_curve_of_GTpd[:, 1] * 100}

                    if self.eval_indoor:
                        res['per_pixel_depth_01'] = mine_recalls_pixel["zeroplane"][2]
                        res['per_pixel_depth_06'] = mine_recalls_pixel["zeroplane"][-1]

                        res['per_plane_depth_005'] = mine_recalls_plane["zeroplane"][1]
                        res['per_plane_depth_01'] = mine_recalls_plane["zeroplane"][2]
                        res['per_plane_depth_06'] = mine_recalls_plane["zeroplane"][-1]

                    else:
                        res['per_pixel_depth_1'] = mine_recalls_pixel["zeroplane"][1]
                        res['per_pixel_depth_10'] = mine_recalls_pixel["zeroplane"][-3]

                        res['per_plane_depth_1'] = mine_recalls_plane["zeroplane"][1]
                        res['per_plane_depth_3'] = mine_recalls_plane["zeroplane"][3]
                        res['per_plane_depth_10'] = mine_recalls_plane["zeroplane"][-3]

                    plot_depth_recall_curve(mine_recalls_pixel, type='pixel (pred_planed vs gt_planed)', save_path=recall_curve_save_path)
                    plot_depth_recall_curve(mine_recalls_plane, type='plane (pred_planed vs gt_planed)', save_path=recall_curve_save_path)

                    # mine_recalls_plane_frompixel = {"zeroplane": self.plane_frompixel_Depth_recall_curve_of_GTpd[:, 0] / self.plane_frompixel_Depth_recall_curve_of_GTpd[:, 1] * 100}

                    # if self.eval_indoor:
                    #     res['per_plane_frompixel_depth_005'] = mine_recalls_plane_frompixel["zeroplane"][1]
                    #     res['per_plane_frompixel_depth_01'] = mine_recalls_plane_frompixel["zeroplane"][2]
                    #     res['per_plane_frompixel_depth_06'] = mine_recalls_plane_frompixel["zeroplane"][-1]

                    # else:
                    #     res['per_plane_frompixel_depth_1'] = mine_recalls_plane_frompixel["zeroplane"][1]
                    #     res['per_plane_frompixel_depth_3'] = mine_recalls_plane_frompixel["zeroplane"][3]
                    #     res['per_plane_frompixel_depth_10'] = mine_recalls_plane_frompixel["zeroplane"][-3]

                    normal_recalls_pixel = {"zeroplane": self.pixelNorm_recall_curve / len(self.RI_VI_SC) * 100}
                    normal_recalls_plane = {"zeroplane": self.planeNorm_recall_curve[:, 0] / self.planeNorm_recall_curve[:, 1] * 100}

                    res['per_pixel_normal_5'] = normal_recalls_pixel["zeroplane"][2]
                    res['per_pixel_normal_30'] = normal_recalls_pixel["zeroplane"][-1]

                    res['per_plane_normal_5'] = normal_recalls_plane["zeroplane"][2]
                    res['per_plane_normal_10'] = normal_recalls_plane["zeroplane"][4]
                    res['per_plane_normal_30'] = normal_recalls_plane["zeroplane"][-1]

                    plot_normal_recall_curve(normal_recalls_pixel, type='pixel', save_path=recall_curve_save_path)
                    plot_normal_recall_curve(normal_recalls_plane, type='plane', save_path=recall_curve_save_path)

                    res["mean_normal_error"] = np.mean(self.bestmatch_normal_errors)
                    res['median_normal_error'] = np.median(self.bestmatch_normal_errors)

        results = OrderedDict({"sem_seg": res})

        if not self.infer_only:
            # file_path = pjoin(self._output_dir, "sem_seg_evaluation.pth")
            # with PathManager.open(file_path, "wb") as f:
            #     torch.save(res, f)

            for k, val in res.items():
                # if 'per_pixel_offset' in k or 'per_plane_offset' in k:
                if 'per_pixel' in k:
                    continue

                print(k, '\t', np.round(val, 2), '\n')

        else:
            print('inference finished...')
            exit(0)

        return results
