import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pose2d_predictions import save_pose_prediction_batch


ORDER = ("body", "left", "right", "face_all")
JOINTS = {"body": 9, "left": 21, "right": 21, "face_all": 18}


def parts(batch, frames, offset):
    result = {}
    for index, part in enumerate(ORDER):
        value = np.ones((batch, frames, JOINTS[part], 3), dtype=np.float32)
        value[..., :2] = offset + index
        result[part] = value
    return result


def test_dump_roundtrip_keeps_native_coords_gt_confidence_and_unequal_lengths(tmp_path):
    pred, ref = parts(2, 8, 2.0), parts(2, 9, 0.5)
    ref["left"][0, 2, :, 2] = 0
    mean = np.linspace(-1, 1, 138, dtype=np.float32)
    std = np.linspace(0.25, 2, 138, dtype=np.float32)
    paths = save_pose_prediction_batch(
        tmp_path, pred, ref, [4, 8], [7, 9], ["clip-a.mp4", "clip-b.mp4"],
        ["first sentence", "second sentence"], mean, std, ORDER, JOINTS,
        back_translations=["first translation", "second translation"],
        metadata={"dataset": "OpenASL", "split": "test"},
    )
    with open(paths[0], "rb") as handle:
        payload = pickle.load(handle)
    assert payload["feats_rst"].shape == (4, 138)
    assert payload["feats_ref"].shape == (7, 138)
    assert payload["conf_ref"].shape == (7, 69)
    raw_xy = np.concatenate([pred[p][0, :4, :, :2].reshape(4, -1)
                             for p in ORDER], axis=-1)
    np.testing.assert_allclose(
        payload["feats_rst"] * payload["motion_std"] + payload["motion_mean"],
        raw_xy, rtol=1e-6, atol=1e-6,
    )
    assert raw_xy.max() > 1  # Saving must not clip to the SLT judge input range.
    assert (payload["conf_ref"][2, 9:30] == 0).all()
    assert payload["name"] == "clip-a.mp4"
    assert payload["text"] == "first sentence"
    assert payload["back_translation"] == "first translation"
    assert not list(tmp_path.glob("*.tmp"))


def test_path_ids_do_not_collide_or_escape_output_directory(tmp_path):
    values = parts(2, 4, 0)
    paths = save_pose_prediction_batch(
        tmp_path, values, values, [4, 4], [4, 4],
        ["group-a/same.mp4", "group-b/same.mp4"], ["a", "b"],
        np.zeros(138), np.ones(138), ORDER, JOINTS,
    )
    assert len(set(paths)) == 2
    assert all(Path(path).parent == tmp_path for path in paths)
    with open(paths[0], "rb") as handle:
        assert pickle.load(handle)["name"] == "group-a/same.mp4"


def test_latent_statistics_are_rejected_as_motion_statistics(tmp_path):
    values = parts(1, 4, 0)
    with unittest.TestCase().assertRaisesRegex(ValueError, "feature width"):
        save_pose_prediction_batch(
            tmp_path, values, values, [4], [4], ["sample"], ["caption"],
            np.zeros(512), np.ones(512), ORDER, JOINTS,
        )
    assert not list(tmp_path.glob("*.pkl"))


class PosePredictionTests(unittest.TestCase):
    def run_case(self, case):
        with tempfile.TemporaryDirectory() as directory:
            case(Path(directory))

    def test_serialized_roundtrip(self):
        self.run_case(test_dump_roundtrip_keeps_native_coords_gt_confidence_and_unequal_lengths)

    def test_original_sample_ids(self):
        self.run_case(test_path_ids_do_not_collide_or_escape_output_directory)

    def test_normalization_width(self):
        self.run_case(test_latent_statistics_are_rejected_as_motion_statistics)


if __name__ == "__main__":
    unittest.main()
