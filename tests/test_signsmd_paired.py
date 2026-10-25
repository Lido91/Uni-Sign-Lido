from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch import nn

from datasets_signsmd import S2T_Dataset_SignSMD
from models_signsmd_paired import Uni_Sign_SignSMD_Paired
from signsmd2d import PARTS_VERSION
from signsmd2d.realization_parts import PARTS


class SignSMDPairedTest(unittest.TestCase):
    @staticmethod
    def _bare_model(part_dim=2):
        model = Uni_Sign_SignSMD_Paired.__new__(Uni_Sign_SignSMD_Paired)
        nn.Module.__init__(model)
        model.pose_part_dim = part_dim
        model.signsmd_part_max_tokens = 16
        model.text_event_dropout = 0.0
        model.mask_text_at_eval = False
        model.signsmd_text_proj = nn.Linear(3, part_dim, bias=False)
        return model

    def test_scatter_is_part_and_span_aligned(self):
        model = self._bare_model()
        vectors = {
            "body a": torch.tensor([2.0, 0.0]),
            "body b": torch.tensor([0.0, 2.0]),
            "left a": torch.tensor([1.0, 1.0]),
        }

        def fake_vectors(this, phrases, device):
            return torch.stack([vectors[phrase] for phrase in phrases]).to(device)

        model._tokenize_event_vectors = types.MethodType(fake_vectors, model)
        model.eval()
        src_input = {
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "signsmd_part_texts": {part: [part] for part in PARTS},
            "signsmd_events": [[
                {"part": "body", "phrase": "body a", "start": 1, "end": 3},
                {"part": "body", "phrase": "body b", "start": 2, "end": 4},
                {"part": "left", "phrase": "left a", "start": 0, "end": 0},
            ]],
        }
        features = model.build_part_text_features(src_input)

        expected_body = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ])
        self.assertTrue(torch.equal(features["body"][0], expected_body))
        self.assertTrue(torch.equal(features["left"][0, 0], vectors["left a"]))
        self.assertEqual(float(features["left"][0, 1:].abs().sum()), 0.0)
        self.assertEqual(float(features["right"].abs().sum()), 0.0)
        self.assertEqual(float(features["face_all"].abs().sum()), 0.0)

        model.text_event_dropout = 1.0
        model.train()
        dropped = model.build_part_text_features(src_input)
        for part in PARTS:
            self.assertEqual(float(dropped[part].abs().sum()), 0.0)

    def test_zero_gate_is_identity_but_final_gate_gets_gradient(self):
        model = self._bare_model()
        model.signsmd_text_gate = nn.Sequential(
            nn.Conv1d(4, 2, 1),
            nn.GELU(),
            nn.Conv1d(2, 1, 1),
            nn.Tanh(),
        )
        nn.init.constant_(model.signsmd_text_gate[0].weight, 0.25)
        nn.init.constant_(model.signsmd_text_gate[0].bias, 0.1)
        nn.init.constant_(model.signsmd_text_gate[2].weight, 0.0)
        nn.init.constant_(model.signsmd_text_gate[2].bias, 0.0)

        pose = torch.ones(1, 2, 4, 3)
        text = torch.ones(1, 4, 2)
        fused = model._inject_part_text("body", pose, text)
        self.assertTrue(torch.equal(fused, pose))

        fused.sum().backward()
        gradient = model.signsmd_text_gate[2].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_event_contract_checks_txt_json_and_pose_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_root = root / "events"
            event_root.mkdir()
            clip = "sample"
            event = {
                "part": "right",
                "kind": "transition",
                "phrase": "right hand: the index finger curls",
                "start": 2,
                "end": 4,
            }
            payload = {
                "contract": PARTS_VERSION,
                "clip": clip,
                "num_steps": 6,
                "events": [event],
            }
            (event_root / f"{clip}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            part_texts = {part: "none" for part in PARTS}
            part_texts["right"] = (
                "right hand: the index finger curls [step 2-4]"
            )

            dataset = S2T_Dataset_SignSMD.__new__(S2T_Dataset_SignSMD)
            dataset.signsmd_event_root = str(event_root)
            dataset.events_required = True
            loaded = dataset._load_signsmd_events(
                f"{clip}.mp4", 6, part_texts
            )
            self.assertEqual(loaded, [event])

            with self.assertRaisesRegex(ValueError, "step mismatch"):
                dataset._load_signsmd_events(f"{clip}.mp4", 5, part_texts)


if __name__ == "__main__":
    unittest.main()
