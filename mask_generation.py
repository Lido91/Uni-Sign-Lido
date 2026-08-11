# Ported from MoMADiff models/mask_transformer/auto_regressive.py
# (MaskGeneration) plus the helpers it depends on from
# models/mask_transformer/tools.py (lengths_to_mask, uniform,
# cosine_schedule). Used by the unified model's MAR-style masked generation.
import math

import torch


# return mask where padding is FALSE
def lengths_to_mask(lengths, max_len):
    '''
    :param lengths: (b, ), length of each sequence
    :param max_len: int, max length of sequence
    :return: mask (b, max_len) where padding is FALSE
    '''
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask  # (b, len)


def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


# More on large value, less on small
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


class MaskGeneration(object):
    def __init__(self, lengths, max_len, num_steps=16, noise_schedule=None,
                 device='cuda', drop_last=True, ignored_position=None,
                 permutation_mode='random'):
        assert len(lengths.shape) == 1
        self.lengths = lengths
        self.max_len = max_len
        self.noise_schedule = noise_schedule
        self.num_steps = num_steps
        self.n_batch = len(lengths)
        self.current_step = 0

        # By creating an extra step if drop_last is True, the last step is
        # ignored in the __next__ method.
        if drop_last:
            num_steps += 1

        self.perm = torch.zeros((self.n_batch, max_len), device=device, dtype=torch.long)
        for i, length in enumerate(lengths):
            # Larger values will be generated at the first iteration
            if permutation_mode == 'random':
                self.perm[i, :length] = torch.randperm(length, device=device)
            elif permutation_mode == 'linear':
                self.perm[i, :length] = torch.arange(length - 1, -1, -1, device=device)
            else:
                raise ValueError(
                    f'In MaskGeneration: invalid permutation mode {permutation_mode}')
            self.perm[i, length:] = max_len + 1
            # mark the ignored position
            if ignored_position is not None:
                self.perm[i, ignored_position[i]] = max_len + 1

        if noise_schedule is None:
            schedule = torch.linspace(0, 1, num_steps, device=device).unsqueeze(-1)
        else:
            schedule = self.noise_schedule(
                torch.linspace(0, 1, num_steps, device=device)).unsqueeze(-1)
        self.mask_table = torch.floor(schedule * lengths)

    def __iter__(self):
        return self

    def __next__(self):
        '''
        Generate mask for each token
        '''
        if self.current_step >= self.num_steps:
            raise StopIteration
        mask = (self.perm < self.mask_table[self.current_step].unsqueeze(-1))
        self.current_step += 1
        return mask
