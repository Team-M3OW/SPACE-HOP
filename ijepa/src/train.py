# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import copy
import logging
import sys
import yaml
import cv2
import wandb
import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from PIL import Image

from src.masks.multiblock import JointMaskCollator as JointMBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.utils.tensors import repeat_interleave_batch
from src.helper import load_checkpoint, init_model, init_opt
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 100
checkpoint_freq = 25
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args, resume_preempt=False):
    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # -- DATA
    dataset_name = args['data']['dataset_name']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']
    patch_size = args['mask']['patch_size']
    num_enc_masks = args['mask']['num_enc_masks']
    min_keep = args['mask']['min_keep']
    enc_mask_scale = args['mask']['enc_mask_scale']
    num_pred_masks = args['mask']['num_pred_masks']
    pred_mask_scale = args['mask']['pred_mask_scale']
    aspect_ratio = args['mask']['aspect_ratio']

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    os.makedirs(folder, exist_ok=True)

    world_size, rank = init_distributed()
    if rank > 0:
        logger.setLevel(logging.ERROR)

    if rank == 0:
        wandb.init(project=args['meta'].get('project_name', "I-JEPA"), config=args, name=tag)

    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- INIT
    encoder, predictor = init_model(
        device=device, patch_size=patch_size, crop_size=crop_size,
        pred_depth=pred_depth, pred_emb_dim=pred_emb_dim, model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    mask_collator = JointMBMaskCollator(
        input_size=crop_size, patch_size=patch_size, pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale, aspect_ratio=aspect_ratio, nenc=num_enc_masks,
        npred=num_pred_masks, allow_overlap=allow_overlap, min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size, crop_scale=crop_scale,
        gaussian_blur=args['data']['use_gaussian_blur'],
        horizontal_flip=args['data']['use_horizontal_flip'],
        color_distortion=args['data']['use_color_distortion'],)
    if dataset_name == 'speedplus':
        from src.datasets.speedplus import make_speedplus
        _, unsupervised_loader, unsupervised_sampler = make_speedplus(
                csv_file=root_path, image_root=image_folder, transform=transform,
                batch_size=batch_size, collator=mask_collator, pin_mem=pin_mem,
                training=True, num_workers=num_workers, world_size=world_size,
                rank=rank, drop_last=True, crop_size=crop_size)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    
    ipe = len(unsupervised_loader)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder, predictor=predictor, wd=wd, final_wd=final_wd,
        start_lr=start_lr, ref_lr=lr, final_lr=final_lr, iterations_per_epoch=ipe,
        warmup=warmup, num_epochs=num_epochs, ipe_scale=ipe_scale, use_bfloat16=use_bfloat16)

    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device, r_path=load_path, encoder=encoder, predictor=predictor,
            target_encoder=target_encoder, opt=optimizer, scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step(); wd_scheduler.step(); next(momentum_scheduler); mask_collator.step()

    def save_checkpoint(epoch, loss_avg):
        save_dict = {'encoder': encoder.state_dict(), 'predictor': predictor.state_dict(),
                     'target_encoder': target_encoder.state_dict(), 'opt': optimizer.state_dict(),
                     'scaler': None if scaler is None else scaler.state_dict(), 'epoch': epoch,
                     'loss': loss_avg, 'batch_size': batch_size, 'lr': lr}
        if rank == 0:
            torch.save(save_dict, latest_path)
            if epoch % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=epoch))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Epoch {epoch + 1}')
        unsupervised_sampler.set_epoch(epoch)
        loss_meter, time_meter = AverageMeter(), AverageMeter()

        pbar = tqdm.tqdm(enumerate(unsupervised_loader), total=ipe, dynamic_ncols=True, disable=(rank != 0))

            # 1. Load Data
# Inside your training loop
        for itr, (imgs, masks_enc, masks_pred, metadata) in pbar:
            
            # 1. Move to device
            imgs = imgs.to(device, non_blocking=True)
            m_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            m_pred = [u.to(device, non_blocking=True) for u in masks_pred]            
    # ... rest of your code
            if any(m.numel() == 0 for m in m_enc + m_pred):
                continue

            def train_step():
                _new_lr, _new_wd = scheduler.step(), wd_scheduler.step()

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))
                        
                        # --- CRITICAL FIX ---
                        # If your model uses a CLS token, it's at index 0. 
                        # m_pred indices refer to spatial patches (0-195).
                        # We must strip CLS so that index 0 of h matches index 0 of the mask.
                        if h.size(1) == 197: 
                            h = h[:, 1:, :] 
                        # --------------------

                        h = apply_masks(h, m_pred)
                        
                        B = imgs.shape[0]
                        h = repeat_interleave_batch(h, B, repeat=len(m_enc))
                        return h

                def forward_context():
                    z = encoder(imgs, m_enc)
                    z = predictor(z, m_enc, m_pred)
                    return z

                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                    h_target = forward_target()
                    z_context = forward_context()
                    loss = F.smooth_l1_loss(z_context, h_target)
                    loss = AllReduce.apply(loss)

                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Momentum Update of Target Encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for pq, pk in zip(encoder.parameters(), target_encoder.parameters()):
                        pk.data.mul_(m).add_((1.-m) * pq.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            if rank == 0:
                if itr % log_freq == 0:
                    wandb.log({"batch_loss": loss, "lr": _new_lr, "wd": _new_wd})
                    logger.info(f'[{epoch+1}, {itr:5d}] loss: {loss_meter.avg:.3f} lr: {_new_lr:.2e}')

            assert not np.isnan(loss), 'loss is nan'

        if rank == 0:
            wandb.log({"epoch_loss": loss_meter.avg, "epoch": epoch + 1})
        save_checkpoint(epoch + 1, loss_meter.avg)

    if rank == 0:
        wandb.finish()

if __name__ == "__main__":
    # Standard boilerplate for running with torchrun / distributed
    # Ensure your config/args are loaded here and passed to main()
    # main(args)
    pass