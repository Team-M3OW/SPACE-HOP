# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from logging import getLogger

import numpy as np

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        # Inside MaskCollator.__call__
        if isinstance(batch[b], dict):
            collated_batch.append(batch[b]['pixel_values'])
        else:
            collated_batch.append(batch[b][0])

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred


class JointMaskCollator(object):
    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False
    ):
        super(JointMaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  
        self.allow_overlap = allow_overlap  
        self._itr_counter = Value('i', -1)  

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        
        # Ensure block isn't larger than the grid
        while h >= self.height: h -= 1
        while w >= self.width: w -= 1
        return (max(1, h), max(1, w))

    def _sample_block_mask(self, b_size, acceptable_regions=None, keypoint=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            if keypoint is not None:
                # y, x are coordinates in patch-space [0, self.height)
                y, x = keypoint 
                
                # --- BOUNDARY FIX ---
                # Ensure y_low < y_high and x_low < x_high
                # top is sampled such that top + h is within grid
                y_low = max(0, int(y - h))
                y_high = max(y_low + 1, min(self.height - h, int(y) + 1))
                
                x_low = max(0, int(x - w))
                x_high = max(x_low + 1, min(self.width - w, int(x) + 1))
                
                top = torch.randint(y_low, y_high, (1,))
                left = torch.randint(x_low, x_high, (1,))
            else:
                top = torch.randint(0, self.height - h, (1,))
                left = torch.randint(0, self.width - w, (1,))
            
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            
            mask_indices = torch.nonzero(mask.flatten())
            valid_mask = len(mask_indices) > self.min_keep
            
            if not valid_mask:
                timeout -= 1
                if timeout <= 0:
                    tries += 1
                    timeout = og_timeout
                    # If we can't find a valid mask with keypoint/overlap constraints,
                    # fallback to random sampling without those constraints.
                    if tries > 5: 
                        return self._sample_block_mask(b_size, acceptable_regions=None)
        
        mask_indices = mask_indices.squeeze()
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        return mask_indices, mask_complement

    def __call__(self, batch):
        '''
        Batch is list of tuples: (img_tensor, kpts_tensor, meta_dict)
        '''
        B = len(batch)
        collated_imgs = []
        collated_metadata = []

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        p_size = self._sample_block_size(g, self.pred_mask_scale, self.aspect_ratio)
        e_size = self._sample_block_size(g, self.enc_mask_scale, (1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred, min_keep_enc = self.height * self.width, self.height * self.width

        for b in range(B):
            img, kpts, meta = batch[b]
            collated_imgs.append(img)
            collated_metadata.append(meta)

            # Sample indices for keypoints to guide target masks
            num_kpts = kpts.shape[0]
            inds = np.random.choice(range(num_kpts), self.npred, replace=(num_kpts < self.npred))
            
            masks_p, masks_C = [], []
            for bb in range(self.npred):
                # SpeedPlus keypoints are [X, Y]. Convert to grid [Y, X].
                kp = kpts[inds[bb]] // self.patch_size
                if (0 <= kp[0] < self.width) and (0 <= kp[1] < self.height):
                    # Pass (y, x) to _sample_block_mask
                    mask, mask_C = self._sample_block_mask(p_size, keypoint=(kp[1], kp[0]))
                else:
                    mask, mask_C = self._sample_block_mask(p_size)
                
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            # Acceptable regions for encoder masks (prevent overlap with targets)
            acceptable_regions = None if self.allow_overlap else masks_C
            
            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # Truncate masks to min_keep to allow stacking into tensors
        collated_masks_pred = [[m[:min_keep_pred] for m in m_list] for m_list in collated_masks_pred]
        collated_masks_enc = [[m[:min_keep_enc] for m in m_list] for m_list in collated_masks_enc]

        # Use default_collate for tensors, return list for metadata
        return (
            torch.utils.data.default_collate(collated_imgs),
            torch.utils.data.default_collate(collated_masks_enc),
            torch.utils.data.default_collate(collated_masks_pred),
            collated_metadata)