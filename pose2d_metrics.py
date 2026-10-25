"""Portable SOKE/MoMADiff confidence-masked DTW-MPJPE2D protocol.

Sources: SOKE/mGPT/metrics/t2m2d.py and MoMADiff/utils/pose2d_metrics.py.
Frame cost is mean L2 distance over GT joints with confidence > 0. Each
part has its own DTW alignment; empty GT frames are dropped for that part.
The sequence score is accumulated DTW cost, WITHOUT path-length division.
Dataset scores average over evaluable sequences; hand = (lhand+rhand)/2.
Inputs are native, de-z-normalized coordinates; no extra clipping is applied.
"""
from collections import defaultdict

import numpy as np
import torch


PART_ORDER = ('body', 'left', 'right', 'face_all')
PART_SLICES = {'all': slice(0, 69), 'body': slice(0, 9),
               'lhand': slice(9, 30), 'rhand': slice(30, 51), 'face': slice(51, 69)}


def masked_dtw_cost(rst, ref, conf):
    """SOKE's anti-diagonal recurrence; equivalent to MoMADiff's double loop."""
    conf = np.asarray(conf) > 0
    ref_has_joint = conf.any(axis=1)
    ref = ref[ref_has_joint]
    conf = conf[ref_has_joint]
    if len(rst) == 0 or len(ref) == 0:
        return None
    if not np.isfinite(rst).all() or not np.isfinite(ref[conf]).all():
        raise ValueError('Non-finite observed coordinates in DTW-MPJPE2D')

    nr, ng = len(rst), len(ref)
    costs = np.empty((nr, ng), dtype=np.float64)
    for j in range(ng):
        mask = conf[j]
        delta = rst[:, mask] - ref[j, mask][None]
        costs[:, j] = np.sqrt(np.square(delta).sum(axis=-1)).mean(axis=-1)

    acc = np.full((nr + 1, ng + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    for d in range(nr + ng - 1):
        i = np.arange(max(0, d - ng + 1), min(nr, d + 1))
        j = d - i
        acc[i + 1, j + 1] = costs[i, j] + np.minimum(
            np.minimum(acc[i, j], acc[i, j + 1]), acc[i + 1, j])
    return float(acc[nr, ng])


def pack_pose_parts(parts):
    """Uni-Sign part dict -> native [B,T,138] xy and [B,T,69] confidence."""
    xy = torch.cat([parts[p][..., :2].flatten(2) for p in PART_ORDER], dim=-1)
    conf = torch.cat([parts[p][..., 2] for p in PART_ORDER], dim=-1)
    return xy, conf


class DTWMPJPE2DMetrics:
    """Rank-local accumulator; same keys and aggregation as SOKE TM2T2DMetrics."""
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
        self.name2scores = defaultdict(dict)

    @torch.no_grad()
    def update(self, feats_rst, feats_ref, conf_ref, lengths, lengths_rst, src, name):
        batch = len(lengths)
        if feats_rst.shape[0] != batch or feats_ref.shape[0] != batch:
            raise ValueError('DTW batch and length counts differ')
        if feats_rst.shape[-1] != 138 or feats_ref.shape[-1] != 138 or conf_ref.shape[-1] != 69:
            raise ValueError('Uni-Sign DTW expects 138D xy features and 69 GT confidences')
        rst_all = feats_rst.detach().cpu().float().reshape(batch, -1, 69, 2).numpy()
        ref_all = feats_ref.detach().cpu().float().reshape(batch, -1, 69, 2).numpy()
        conf_all = (conf_ref.detach().cpu().float() > 0).numpy()
        for i in range(batch):
            source = str(src[i]).lower()
            if source not in ('how2sign', 'openasl'):
                raise ValueError(f'Unsupported DTW source: {source}')
            nr, ng = int(lengths_rst[i]), int(lengths[i])
            if not 0 <= nr <= rst_all.shape[1] or not 0 <= ng <= min(ref_all.shape[1], conf_all.shape[1]):
                raise ValueError('DTW true length exceeds available frames')
            rst, ref, conf = rst_all[i, :nr], ref_all[i, :ng], conf_all[i, :ng]
            for part, sl in PART_SLICES.items():
                value = masked_dtw_cost(rst[:, sl], ref[:, sl], conf[:, sl])
                if value is not None:
                    key = f'{source}_DTW_MPJPE2D_{part}'
                    self.sums[key] += value
                    self.counts[key] += 1
                    self.name2scores[str(name[i])][key] = value

    def update_parts(self, pred_parts, gt_parts, gt_lengths, pred_lengths, dataset, names):
        rst, _ = pack_pose_parts(pred_parts)
        ref, conf = pack_pose_parts(gt_parts)
        self.update(rst, ref, conf, gt_lengths, pred_lengths,
                    [dataset.lower()] * len(gt_lengths), names)

    def results(self):
        result = {key: self.sums[key] / count for key, count in self.counts.items() if count}
        for source in ('how2sign', 'openasl'):
            left, right = f'{source}_DTW_MPJPE2D_lhand', f'{source}_DTW_MPJPE2D_rhand'
            if left in result and right in result:
                result[f'{source}_DTW_MPJPE2D_hand'] = (result[left] + result[right]) / 2
        return result
