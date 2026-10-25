"""Merged indexing/collation and source-specific checkpoint-selection checks."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import RandomSampler

from datasets import Base_Dataset
from datasets_merged import MergedSignDataset, build_merged_training_dataset
from train_unified import (evaluate_merged_validation, get_unified_args_parser,
                           validate_args)


class ChildFixture(Base_Dataset):
    rgb_support = False

    def __init__(self, count, name):
        self.count, self.name = count, name

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        counts = {'body': 9, 'left': 21, 'right': 21, 'face_all': 18}
        pose = {p: torch.ones(3 + index, n, 3) for p, n in counts.items()}
        return str(index), pose, self.name + str(index), '', {}


class MergedDatasetTests(unittest.TestCase):
    def test_concat_keeps_natural_counts_and_namespaces_ids(self):
        merged = MergedSignDataset([ChildFixture(3, 'a'), ChildFixture(1, 'b')])
        self.assertEqual(len(merged), 4)
        self.assertEqual(merged.source_counts, {'OpenASL': 3, 'How2Sign': 1})
        names = [merged[i][0] for i in RandomSampler(merged)]
        self.assertEqual(len(set(names)), 4)
        self.assertEqual(sum(n.startswith('OpenASL/') for n in names), 3)
        self.assertEqual(sum(n.startswith('How2Sign/') for n in names), 1)
        self.assertEqual(merged[-1][0], 'How2Sign/0')
        with self.assertRaises(IndexError):
            merged[4]

    def test_mixed_collation_preserves_pose_lengths_and_source(self):
        merged = MergedSignDataset([ChildFixture(3, 'a'), ChildFixture(1, 'b')])
        src, tgt = merged.collate_fn([merged[2], merged[3]])
        self.assertEqual(src['src_length_batch'].tolist(), [5, 3])
        self.assertEqual(src['name_batch'], ['OpenASL/2', 'How2Sign/0'])
        self.assertEqual(src['dataset_name_batch'], ['OpenASL', 'How2Sign'])
        self.assertEqual(tgt['gt_sentence'], ['a2', 'b0'])
        self.assertEqual(src['body'].shape, (2, 5, 9, 3))
        self.assertEqual(src['attention_mask'].tolist(), [[1]*5, [1, 1, 1, 0, 0]])

    def test_child_args_keep_correct_source_roots_and_frame_caps(self):
        args = SimpleNamespace(dataset='How2Sign', max_length=256, t2p_max_length=400)
        calls = []
        def factory(path, child_args, phase):
            calls.append((path, child_args.dataset, child_args.max_length, phase))
            return ChildFixture(1, child_args.dataset)
        with patch('datasets_merged.S2T_Dataset', side_effect=factory):
            build_merged_training_dataset(args)
        with patch('datasets_merged.S2T_Dataset_Pose2D_KLVAE', side_effect=factory):
            build_merged_training_dataset(args, generation=True)
        self.assertEqual([c[1:] for c in calls], [
            ('OpenASL',256,'train'), ('How2Sign',256,'train'),
            ('OpenASL',400,'train'), ('How2Sign',400,'train')])
        self.assertEqual(args.dataset, 'How2Sign')
        self.assertEqual(args.max_length, 256)

    def test_validation_reports_both_and_selects_explicit_primary(self):
        args = SimpleNamespace(dataset='How2Sign', skip_slg_generation_val=False)
        loaders = {name: (SimpleNamespace(dataset=name), name+'-slt', name+'-slg')
                   for name in ('OpenASL','How2Sign')}
        def slt(cfg, loader, model, raw, phase):
            self.assertEqual(loader, cfg.dataset+'-slt')
            self.assertEqual(phase, 'test')
            return {'bleu4': 10.0 if cfg.dataset=='OpenASL' else 20.0}
        def slg(cfg, loader, raw):
            self.assertEqual(loader, cfg.dataset+'-slg')
            return {cfg.dataset.lower()+'_DTW_MPJPE2D_hand': 0.2}
        with patch('train_unified.evaluate', side_effect=slt), \
             patch('train_unified.evaluate_t2p_loss', return_value={'t2p_loss':1.0}), \
             patch('train_unified.evaluate_slg_generation', side_effect=slg):
            stats = evaluate_merged_validation(args, loaders, object(), SimpleNamespace())
        self.assertEqual(stats['test/openasl/slt_bleu4'], 10)
        self.assertEqual(stats['test/how2sign/slt_bleu4'], 20)
        self.assertEqual(stats['slt_bleu4'], 20)
        self.assertIn('how2sign_DTW_MPJPE2D_hand',stats)

    def test_merged_cli_requires_joint_training_and_test(self):
        base = ['--vae_checkpoint','unused','--merge_datasets','--dataset','How2Sign']
        parser = get_unified_args_parser()
        validate_args(parser.parse_args(base+['--train_stage','3']))
        with self.assertRaisesRegex(ValueError,'Merged'):
            validate_args(parser.parse_args(base+['--train_stage','1']))
        with self.assertRaisesRegex(ValueError,'Merged'):
            validate_args(parser.parse_args(base+['--train_stage','3','--val_split','dev']))


if __name__ == '__main__':
    unittest.main()
