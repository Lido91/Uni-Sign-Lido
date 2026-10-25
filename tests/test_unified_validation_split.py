"""Validation protocol and resume selection must agree across split changes."""
import unittest
from types import SimpleNamespace

from config import test_label_paths, dev_label_paths
from train_unified import get_unified_args_parser, resolve_validation_split, restore_best_metric


class ValidationSplitTests(unittest.TestCase):
    def test_default_is_test(self):
        args = get_unified_args_parser().parse_args(['--vae_checkpoint', 'unused'])
        self.assertEqual(args.val_split, 'test')

    def test_test_labels_for_both_datasets(self):
        for dataset in ('OpenASL', 'How2Sign'):
            self.assertEqual(resolve_validation_split(SimpleNamespace(
                dataset=dataset, val_split='test')), ('test', test_label_paths[dataset]))

    def test_explicit_openasl_dev(self):
        self.assertEqual(resolve_validation_split(SimpleNamespace(
            dataset='OpenASL', val_split='dev')), ('dev', dev_label_paths['OpenASL']))

    def test_missing_dev_does_not_silently_use_test(self):
        with self.assertRaisesRegex(ValueError, 'No dev labels'):
            resolve_validation_split(SimpleNamespace(dataset='How2Sign', val_split='dev'))

    def test_legacy_openasl_dev_best_resets_on_test(self):
        self.assertEqual(restore_best_metric({'best_metric': 24}, 'OpenASL', 'test'), float('-inf'))
        self.assertEqual(restore_best_metric({'best_metric': 24}, 'OpenASL', 'dev'), 24)

    def test_legacy_how2sign_test_best_is_retained(self):
        self.assertEqual(restore_best_metric({'best_metric': 14}, 'How2Sign', 'test'), 14)

    def test_explicit_checkpoint_protocol(self):
        state = {'best_metric': 23, 'validation_split': 'test'}
        self.assertEqual(restore_best_metric(state, 'OpenASL', 'test'), 23)
        self.assertEqual(restore_best_metric(state, 'OpenASL', 'dev'), float('-inf'))


if __name__ == '__main__':
    unittest.main()
