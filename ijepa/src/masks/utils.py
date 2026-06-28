# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch

def apply_masks(x, masks, has_cls=False):
    all_x = []
    for m in masks:
        if has_cls:
            m = m + 1
            
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
        
    return torch.cat(all_x, dim=0)