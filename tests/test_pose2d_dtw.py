import unittest

import numpy as np
import torch

from pose2d_metrics import DTWMPJPE2DMetrics, masked_dtw_cost, pack_pose_parts


class DTWProtocolTests(unittest.TestCase):
    def test_different_speed_aligns_to_zero(self):
        rst = np.array([0, 1, 2], dtype=np.float32)[:, None, None] * np.array([1, 0])
        ref = np.array([0, 0, 1, 2], dtype=np.float32)[:, None, None] * np.array([1, 0])
        self.assertEqual(masked_dtw_cost(rst, ref, np.ones((4, 1))), 0)

    def test_accumulated_cost_is_not_path_mean(self):
        rst = np.ones((4, 1, 2), dtype=np.float32)
        rst[..., 1] = 0
        self.assertEqual(masked_dtw_cost(rst, np.zeros_like(rst), np.ones((4, 1))), 4)

    def test_missing_joints_and_frames_are_ignored(self):
        rst = np.zeros((2, 2, 2), dtype=np.float32)
        ref = np.zeros((3, 2, 2), dtype=np.float32)
        ref[:, 1] = 1000
        ref[1, 0] = 1000
        conf = np.array([[1, 0], [0, 0], [1, 0]])
        self.assertEqual(masked_dtw_cost(rst, ref, conf), 0)
        self.assertIsNone(masked_dtw_cost(rst, ref, np.zeros_like(conf)))

    def test_unequal_lengths_use_full_reference(self):
        rst = np.zeros((2, 1, 2), dtype=np.float32)
        ref = np.zeros((3, 1, 2), dtype=np.float32)
        ref[-1, 0, 0] = 2
        self.assertEqual(masked_dtw_cost(rst, ref, np.ones((3, 1))), 2)

    def test_sequence_mean_hand_mean_and_padding(self):
        pred = torch.zeros(2, 4, 69, 2)
        ref = torch.zeros_like(pred)
        pred[:, :, 9:30, 0] = 1
        pred[:, :, 30:51, 0] = 3
        pred[0, 2:] = 1000  # excluded by the true generated length
        metric = DTWMPJPE2DMetrics()
        metric.update(pred.flatten(2), ref.flatten(2), torch.ones(2, 4, 69),
                      [2, 4], [2, 4], ['openasl', 'openasl'], ['short', 'long'])
        results = metric.results()
        self.assertEqual(results['openasl_DTW_MPJPE2D_lhand'], 3)
        self.assertEqual(results['openasl_DTW_MPJPE2D_rhand'], 9)
        self.assertEqual(results['openasl_DTW_MPJPE2D_hand'], 6)
        self.assertEqual(metric.counts['openasl_DTW_MPJPE2D_all'], 2)

    def test_part_packing_preserves_native_coordinates(self):
        parts = {p: torch.zeros(1, 2, j, 3)
                 for p, j in [('body', 9), ('left', 21), ('right', 21), ('face_all', 18)]}
        parts['left'][..., 0] = 2
        for part in parts.values():
            part[..., 2] = 1
        feats, conf = pack_pose_parts(parts)
        self.assertEqual(feats.shape[-1], 138)
        self.assertEqual(feats[0, 0, 18].item(), 2)
        metric = DTWMPJPE2DMetrics()
        metric.update(feats, torch.zeros_like(feats), conf, [2], [2], ['how2sign'], ['raw'])
        self.assertEqual(metric.results()['how2sign_DTW_MPJPE2D_lhand'], 4)

    def test_nonfinite_observed_prediction_fails(self):
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            masked_dtw_cost(np.full((1, 1, 2), np.nan), np.zeros((1, 1, 2)), np.ones((1, 1)))


if __name__ == '__main__':
    unittest.main()
