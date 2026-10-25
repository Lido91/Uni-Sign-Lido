"""Verify exact source reuse, real geometric gradients, and optimizer recovery."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import MT5Config, MT5ForConditionalGeneration

from models_unified import Uni_Sign_Unified
from models_unified_geosign import Uni_Sign_Unified_GeoSign, GEOSIGN_PREFIXES
from vendor.geosign import models as reference
from geosign_training import (create_geosign_optimizers, restore_geosign_optimizer,
                              step_geosign)
from train_unified import (apply_stage_freezing, unified_state_dict, load_partial,
                           VAE_OWNED_PREFIXES)

ROOT = Path(__file__).resolve().parents[1]


def fixture_args():
    return SimpleNamespace(
        hyp_dim=8, init_c=1.0, label_smoothing_hyp=0.2,
        hyp_text_cmp='token', hyp_text_emb_src='decoder', total_steps=20,
        alpha=0.7, label_smoothing=0.2, eval=False,
        lr=1e-4, weight_decay=1e-4, opt_betas=None,
        hyp_lr=0.001, hyp_stabilize=100,
    )


def tiny_parent_init(self, args):
    nn.Module.__init__(self)
    self.args, self.lang, self.default_task = args, 'English', 'slt'
    self.modes = ['body', 'left', 'right', 'face_all']
    self.proj_linear = nn.ModuleDict({p: nn.Linear(3, 4) for p in self.modes})
    self.gcn_modules = nn.ModuleDict({p: nn.Identity() for p in self.modes})
    self.fusion_gcn_modules = nn.ModuleDict({p: nn.Identity() for p in self.modes})
    self.part_para = nn.Parameter(torch.zeros(16))
    self.pose_proj = nn.Linear(16, 32)
    self.mt5_model = MT5ForConditionalGeneration(MT5Config(
        vocab_size=32, d_model=32, d_ff=64, num_heads=4, d_kv=8,
        num_layers=1, num_decoder_layers=1, dropout_rate=0.0,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    ))
    self.mt5_tokenizer = SimpleNamespace(pad_token_id=0)
    self.vae_encoder, self.vae_decoder = nn.Linear(2, 2), nn.Linear(2, 2)
    self.latent_in_proj = nn.Linear(2, 32)
    self.mask_emb = nn.Parameter(torch.zeros(32))
    self.cond_out_head, self.diff_head = nn.Linear(32, 2), nn.Linear(2, 2)


def tiny_t2p(self, src, tgt):
    hidden = self.mt5_model.encoder(
        inputs_embeds=self.latent_in_proj(src['latent']) + self.mask_emb,
        return_dict=True,
    ).last_hidden_state
    return {'loss': self.diff_head(self.cond_out_head(hidden)).square().mean()}


def make_fixture():
    with patch.object(Uni_Sign_Unified, '__init__', tiny_parent_init):
        return Uni_Sign_Unified_GeoSign(fixture_args())


def fixture_batch(device='cpu'):
    counts = {'body': 9, 'left': 21, 'right': 21, 'face_all': 18}
    src = {p: torch.randn(2, 4, n, 3, device=device) * 0.1
           for p, n in counts.items()}
    src.update(attention_mask=torch.ones(2, 4, dtype=torch.long, device=device),
               prefix_ids=torch.tensor([[2, 3], [2, 3]], device=device),
               prefix_mask=torch.ones(2, 2, dtype=torch.long, device=device))
    tgt = {'labels_ids': torch.tensor([[4, 5, 1, 0], [6, 7, 8, 1]], device=device),
           'gt_sentence': ['a b', 'c d']}
    gen = {'latent': torch.randn(2, 3, 2, device=device)}
    return src, tgt, gen


class GeosignUnifiedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_source_is_byte_identical_to_recorded_origin(self):
        manifest = json.loads((ROOT / 'vendor/geosign/provenance.json').read_text())
        expected = next(iter(manifest['copied_model_sha256'].values()))
        self.assertEqual(hashlib.sha256((ROOT / 'vendor/geosign/models.py').read_bytes()).hexdigest(), expected)
        source = Path(manifest['local_source']) / 'models.py'
        if source.exists():
            self.assertEqual((ROOT / 'vendor/geosign/models.py').read_bytes(), source.read_bytes())
        def init_block(path, classname):
            tree = ast.parse(path.read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
            init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
            node = next(n for n in init.body if isinstance(n, ast.If)
                        and ast.unparse(n.test) == 'self.use_hyp and GEOOPT_AVAILABLE')
            return ast.dump(node, include_attributes=False)
        self.assertEqual(init_block(ROOT / 'vendor/geosign/models.py', 'Uni_Sign'),
                         init_block(ROOT / 'models_unified_geosign.py', 'Uni_Sign_Unified_GeoSign'))

    def test_forward_reuses_exact_reference_and_has_finite_gradients(self):
        module = make_fixture()
        src, tgt, _ = fixture_batch()
        expected = reference.Uni_Sign.forward(module, src, tgt)
        actual = module(src, tgt)
        for key in ('loss', 'ce_loss', 'margin_loss', 'alpha', 'curvature'):
            torch.testing.assert_close(actual[key], expected[key])
        torch.testing.assert_close(actual['alpha'], torch.tensor(0.8))
        actual['loss'].backward()
        for parameter in (module.hyp_attn_b, module.hyp_proj_text.proj.weight,
                          module.manifold.isp_c, module.geom_loss.temp):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(module.hyp_proj_text.proj.weight.grad.abs().sum().item(), 0)

    def test_joint_objective_preserves_diffusion_and_stage_freezing(self):
        module = make_fixture()
        apply_stage_freezing(module, 2)
        self.assertFalse(module.pose_proj.weight.requires_grad)
        self.assertTrue(module.hyp_attn_b.requires_grad)
        self.assertTrue(module.mt5_model.shared.weight.requires_grad)
        self.assertFalse(module.vae_encoder.weight.requires_grad)
        module.eval()
        src, tgt, gen = fixture_batch()
        with patch.object(Uni_Sign_Unified, 'forward_t2p', tiny_t2p):
            slt = module(src, tgt, task='slt')
            t2p = module(gen, tgt, task='t2p')
            joint = module(src, tgt, task='joint', t2p_src_input=gen,
                           t2p_tgt_input=tgt, slt_weight=2, t2p_weight=3)
        torch.testing.assert_close(joint['loss'], 2 * slt['loss'] + 3 * t2p['loss'])
        apply_stage_freezing(module, 3)
        self.assertTrue(module.pose_proj.weight.requires_grad)
        self.assertFalse(module.vae_decoder.weight.requires_grad)

    def test_two_optimizers_step_and_restore_geometry_state(self):
        module = make_fixture()
        apply_stage_freezing(module, 3)
        optimizer, hyp = create_geosign_optimizers(fixture_args(), module)
        euclid_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
        hyp_ids = {id(p) for g in hyp.param_groups for p in g['params']}
        self.assertFalse(euclid_ids & hyp_ids)
        self.assertIn(id(module.hyp_attn_b), hyp_ids)
        self.assertIn(id(module.manifold.isp_c), euclid_ids)
        src, tgt, _ = fixture_batch()
        before = module.hyp_attn_b.detach().clone()
        module(src, tgt)['loss'].backward()
        hyp.step(); hyp.zero_grad(set_to_none=True); optimizer.step()
        self.assertFalse(torch.equal(before, module.hyp_attn_b))
        state = copy.deepcopy(hyp.state_dict())
        _, restored = create_geosign_optimizers(fixture_args(), module)
        restore_geosign_optimizer({'hyp_optimizer': state}, restored)
        self.assertEqual(restored.state_dict()['param_groups'], state['param_groups'])
        for key, values in state['state'].items():
            for name, value in values.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(restored.state_dict()['state'][key][name], value)
        with self.assertRaisesRegex(ValueError, 'lacks'):
            restore_geosign_optimizer({}, restored)

    def test_plain_checkpoint_requires_explicit_geometry_initialization(self):
        module = make_fixture()
        plain = {k: v for k, v in unified_state_dict(module).items()
                 if not k.startswith(GEOSIGN_PREFIXES)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'plain.pth'
            torch.save({'model': plain}, path)
            with self.assertRaisesRegex(ValueError, 'required keys'):
                load_partial(module, path, 'enhanced', ())
            load_partial(module, path, 'plain initialization', VAE_OWNED_PREFIXES + GEOSIGN_PREFIXES)
            bad = dict(plain)
            del bad['diff_head.weight']
            torch.save({'model': bad}, path)
            with self.assertRaisesRegex(ValueError, 'required keys'):
                load_partial(module, path, 'bad generation checkpoint', VAE_OWNED_PREFIXES + GEOSIGN_PREFIXES)

    def test_accumulation_boundary_and_nonfinite_update(self):
        module = make_fixture()
        apply_stage_freezing(module, 3)
        optimizer, hyp = create_geosign_optimizers(fixture_args(), module)

        class Engine:
            def __init__(self):
                self.module = nn.Module()
                self.module.add_module('module', module)
                self.optimizer = optimizer
                self.micro_steps = self.skipped_steps = 0
                self.hyp_cleared_before_update = False

            def is_gradient_accumulation_boundary(self):
                return (self.micro_steps + 1) % 2 == 0

            def step(self):
                if self.is_gradient_accumulation_boundary():
                    self.hyp_cleared_before_update = module.hyp_attn_b.grad is None
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                self.micro_steps += 1

            def zero_grad(self):
                self.module.zero_grad(set_to_none=True)

        engine = Engine()
        src, tgt, _ = fixture_batch()
        before = module.hyp_attn_b.detach().clone()
        (module(src, tgt)['loss'] / 2).backward()
        self.assertTrue(step_geosign(engine, module, hyp))
        self.assertTrue(torch.equal(before, module.hyp_attn_b))
        self.assertEqual(module.global_step.item(), 0)
        (module(src, tgt)['loss'] / 2).backward()
        self.assertTrue(step_geosign(engine, module, hyp))
        self.assertFalse(torch.equal(before, module.hyp_attn_b))
        self.assertTrue(engine.hyp_cleared_before_update)
        self.assertEqual(module.global_step.item(), 1)
        before = module.hyp_attn_b.detach().clone()
        (module(src, tgt)['loss'] / 2).backward()
        step_geosign(engine, module, hyp)
        (module(src, tgt)['loss'] / 2).backward()
        module.hyp_attn_b.grad.fill_(float('nan'))
        self.assertFalse(step_geosign(engine, module, hyp))
        self.assertTrue(torch.equal(before, module.hyp_attn_b))
        self.assertEqual(module.global_step.item(), 1)
        self.assertEqual(engine.skipped_steps, 1)


if __name__ == '__main__':
    unittest.main()
