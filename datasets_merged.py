"""Natural concatenation of OpenASL and How2Sign, without moving pose files."""
import bisect
import copy

from torch.utils.data import ConcatDataset

from config import train_label_paths
from datasets import S2T_Dataset
from datasets_pose2d_klvae import S2T_Dataset_Pose2D_KLVAE

MERGED_DATASETS = ('OpenASL', 'How2Sign')


class MergedSignDataset(ConcatDataset):
    """One index space, one ordinary shuffled sampler, natural source proportions.

    Each child retains its own pose root and preprocessing. Sample IDs are
    namespaced, so identical IDs from different corpora cannot collide.
    """
    def __init__(self, datasets, names=MERGED_DATASETS):
        if len(datasets) != len(names) or not datasets:
            raise ValueError('Each dataset requires exactly one source name')
        if len(set(names)) != len(names):
            raise ValueError('Merged source names must be unique')
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError('Cannot merge an empty training dataset')
        super().__init__(datasets)
        self.names = tuple(names)
        self.source_counts = {name: len(dataset) for name, dataset in zip(names, datasets)}

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        child = bisect.bisect_right(self.cumulative_sizes, index)
        offset = 0 if child == 0 else self.cumulative_sizes[child - 1]
        sample = self.datasets[child][index - offset]
        return (f'{self.names[child]}/{sample[0]}', *sample[1:])

    def collate_fn(self, batch):
        # Both children use the same pose-only Base_Dataset collation.
        src, tgt = self.datasets[0].collate_fn(batch)
        src['dataset_name_batch'] = [sample[0].split('/', 1)[0] for sample in batch]
        return src, tgt

    def __repr__(self):
        return f'MergedSignDataset(total={len(self)}, sources={self.source_counts})'


def build_merged_training_dataset(args, *, generation=False):
    cls = S2T_Dataset_Pose2D_KLVAE if generation else S2T_Dataset
    children = []
    for name in MERGED_DATASETS:
        child_args = copy.copy(args)
        child_args.dataset = name
        if generation:
            child_args.max_length = args.t2p_max_length
        children.append(cls(train_label_paths[name], child_args, phase='train'))
    return MergedSignDataset(children)
