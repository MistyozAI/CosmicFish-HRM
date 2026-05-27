import os
import time
import math
from contextlib import nullcontext
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import HRMCosmicFish, HRMCosmicFishConfig


def get_batch(data_path, block_size, batch_size, device):
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    device_obj = torch.device(device) if isinstance(device, str) else device
    if device_obj.type == 'cuda':
        x, y = x.pin_memory().to(device_obj, non_blocking=True), y.pin_memory().to(device_obj, non_blocking=True)
    else:
        x, y = x.to(device_obj), y.to(device_obj)
    return x, y


def get_args():
    parser = argparse.ArgumentParser(description='Fine-tune CosmicFish-HRM')

    parser.add_argument('--data_dir', type=str, default='data/alpaca_gpt4_cleaned_pure')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=100000)
    parser.add_argument('--learning_rate', type=float, default=6e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--lr_decay_iters', type=int, default=100000)
    parser.add_argument('--min_lr', type=float, default=None)

    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--eval_iters', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=1)

    parser.add_argument('--out_dir', type=str, default='out_finetune')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--seed', type=int, default=1337)

    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--resume_finetune', type=str, default=None)

    return parser.parse_args()


def get_lr(it, args):
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    if it > args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


def setup_distributed():
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, seed_offset


@torch.no_grad()
def estimate_loss(model, data_dir, args, config, device, ctx):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        data_path = os.path.join(data_dir, f'{split}.bin')
        for k in range(args.eval_iters):
            X, Y = get_batch(data_path, config.block_size, args.batch_size, device)
            with ctx:
                logits, loss, _, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def print_progress(iter_num, max_iters, loss, lr, avg_hrm_steps, val_loss=None):
    percent_done = (iter_num / max_iters) * 100
    progress_str = (
        f"\rIter {iter_num}/{max_iters} ({percent_done:.1f}%) | "
        f"Loss:{loss:.4f} | LR:{lr:.2e} | "
        f"HRM:{avg_hrm_steps:.1f}"
    )
    if val_loss is not None:
        progress_str += f" | Val:{val_loss:.4f}"
    print(progress_str, end='', flush=True)


def main():
    args = get_args()

    if args.min_lr is None:
        args.min_lr = args.learning_rate / 10

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, seed_offset = setup_distributed()

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)

    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    train_path = os.path.join(args.data_dir, 'train.bin')
    val_path   = os.path.join(args.data_dir, 'val.bin')

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise ValueError(f"Training data not found at {args.data_dir}. Run prepare_singleturn.py first!")

    if master_process:
        print(f"Loading fine-tuning data from {args.data_dir}")
        print(f"\nLoading pretrained model from: {args.pretrained_model}")

    checkpoint = torch.load(args.pretrained_model, map_location=device, weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError("No config found in pretrained checkpoint")

    config.dropout = 0.1
    config.hrm_exploration_prob = 0.05

    model = HRMCosmicFish(config)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        raise ValueError("No model weights found in checkpoint")

    cleaned_state_dict = {
        k.replace('module.', '').replace('_orig_mod.', ''): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict)

    if master_process:
        print(f"Pretrained model loaded: {model.get_num_params() / 1e6:.2f}M parameters")
        print(f"  Input blocks: {config.n_input_layers} | HRM: H={config.hrm_H_layers} L={config.hrm_L_layers} | Output blocks: {config.n_output_layers}")

    iter_num = 0
    best_val_loss = float('inf')

    if args.resume_finetune:
        if master_process:
            print(f"Resuming fine-tuning from: {args.resume_finetune}")
        resume_checkpoint = torch.load(args.resume_finetune, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint['model'])
        iter_num = resume_checkpoint['iter_num']
        best_val_loss = resume_checkpoint['best_val_loss']
        if master_process:
            print(f"Resumed from iteration {iter_num}, best val loss: {best_val_loss:.4f}")

    model.to(device)

    if args.compile and hasattr(torch, 'compile'):
        if master_process:
            print("Compiling model...")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)

    if args.resume_finetune and 'optimizer' in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint['optimizer'])

    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    if master_process:
        print(f"\nFine-tuning started | LR: {args.learning_rate:.2e} | Batch: {args.batch_size * args.gradient_accumulation_steps * ddp_world_size} | Max iters: {args.max_iters}\n")

    start_time = time.time()

    while iter_num < args.max_iters:
        lr = get_lr(iter_num, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % args.eval_interval == 0 and master_process:
            losses = estimate_loss(model, args.data_dir, args, config, device, ctx)
            if iter_num > 0:
                print()
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'args': vars(args),
                        'pretrained_from': args.pretrained_model,
                    }
                    torch.save(checkpoint, os.path.join(args.out_dir, 'best_model.pt'))
                    print(f"Saved best model (val_loss={best_val_loss:.4f})")

        if iter_num % args.save_interval == 0 and iter_num > 0 and master_process:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'args': vars(args),
                'pretrained_from': args.pretrained_model,
            }
            torch.save(checkpoint, os.path.join(args.out_dir, f'ckpt_{iter_num:06d}.pt'))
            print(f"\nSaved checkpoint at iteration {iter_num}")

        for micro_step in range(args.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)

            data_path = os.path.join(args.data_dir, 'train.bin')
            X, Y = get_batch(data_path, config.block_size, args.batch_size, device)

            with ctx:
                logits, loss, steps_taken, q_logits = model(X, Y)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if master_process and (iter_num % args.log_interval == 0):
            lossf     = loss.item() * args.gradient_accumulation_steps
            avg_steps = steps_taken.float().mean().item()
            print_progress(iter_num, args.max_iters, lossf, lr, avg_steps)

        iter_num += 1

    if master_process:
        print()
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'args': vars(args),
            'pretrained_from': args.pretrained_model,
        }
        torch.save(checkpoint, os.path.join(args.out_dir, 'final_finetuned_model.pt'))
        print(f"Fine-tuning complete | Final: {args.out_dir}/final_finetuned_model.pt | Best val loss: {best_val_loss:.4f}")

    if ddp:
        dist.destroy_process_group()


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    decay   = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    param_dict  = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))    if pn in param_dict], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict], "weight_decay": 0.0},
    ]

    use_fused = (device_type == 'cuda') and ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames)
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **(dict(fused=True) if use_fused else {}))
    return optimizer


HRMCosmicFish.configure_optimizers = configure_optimizers

if __name__ == '__main__':
    main()