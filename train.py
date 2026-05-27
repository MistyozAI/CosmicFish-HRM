import os
import sys
import time
import math
from contextlib import nullcontext
import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

WANDB_API_KEY = "wandb key here!!"
WANDB_ENTITY  = "more wandb stuff"

os.environ["WANDB_API_KEY"] = WANDB_API_KEY

try:
    import wandb
except ImportError:
    print("[ERROR] wandb is not installed. Run: pip install wandb")
    sys.exit(1)

try:
    result = wandb.login(key=WANDB_API_KEY, relogin=True, force=True)
    if not result:
        raise RuntimeError("wandb.login() returned False — key rejected.")
except Exception as e:
    print(f"[ERROR] wandb login failed: {e}")
    sys.exit(1)

from model import HRMCosmicFish, HRMCosmicFishConfig


def get_batch(data_path, block_size, batch_size, device):
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    device_obj = torch.device(device) if isinstance(device, str) else device
    if device_obj.type == 'cuda':
        x = x.pin_memory().to(device_obj, non_blocking=True)
        y = y.pin_memory().to(device_obj, non_blocking=True)
    else:
        x, y = x.to(device_obj), y.to(device_obj)
    return x, y


def get_args():
    parser = argparse.ArgumentParser(description='Train CosmicFish-HRM')

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset_names', type=str, nargs='+', default=None)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=50000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--warmup_iters', type=int, default=2000)
    parser.add_argument('--lr_decay_iters', type=int, default=100000)
    parser.add_argument('--min_lr', type=float, default=3e-5)

    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=5000)

    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--seed', type=int, default=1337)

    parser.add_argument('--init_from', type=str, default='scratch',
                        choices=['scratch', 'resume'])
    parser.add_argument('--resume_path', type=str, default=None)

    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='CosmicFish-HRM')
    parser.add_argument('--wandb_run_name', type=str, default=None)

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
def estimate_loss(model, data_dir, dataset_names, args, device, ctx):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        hrm_steps_all = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            if dataset_names:
                dataset_name = np.random.choice(dataset_names)
                data_path = os.path.join(data_dir, dataset_name, f'{split}.bin')
            else:
                data_path = os.path.join(data_dir, f'{split}.bin')
            X, Y = get_batch(data_path, args.block_size_eval, args.batch_size, device)
            with ctx:
                logits, loss, steps_taken, q_logits = model(X, Y)
            losses[k] = loss.item()
            hrm_steps_all[k] = steps_taken.float().mean().item()
        out[f'{split}_loss'] = losses.mean().item()
        out[f'{split}_hrm_steps'] = hrm_steps_all.mean().item()
    model.train()
    return out


def save_training_plot(history, out_dir):
    if not history['iters']:
        return

    fig = plt.figure(figsize=(18, 10), facecolor='#0f0f0f')
    fig.suptitle('CosmicFish-HRM — Training Dashboard', color='white',
                 fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor('#1a1a1a')
        ax.set_title(title, color='white', fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, color='#aaaaaa', fontsize=9)
        ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=9)
        ax.tick_params(colors='#777777')
        ax.spines[:].set_color('#333333')
        ax.grid(color='#2a2a2a', linestyle='--', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'Training Loss', 'Iteration', 'Loss')
    if history['train_loss']:
        ax1.plot(history['loss_iters'], history['train_loss'],
                 color='#4fc3f7', linewidth=1.2, alpha=0.9, label='train loss')
        ax1.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'Validation Loss', 'Iteration', 'Loss')
    if history['val_loss']:
        ax2.plot(history['eval_iters'], history['val_loss'],
                 color='#ef5350', linewidth=1.5, marker='o', markersize=3, label='val loss')
        ax2.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, 'Loss Comparison', 'Iteration', 'Loss')
    if history['train_loss']:
        ax3.plot(history['loss_iters'], history['train_loss'],
                 color='#4fc3f7', linewidth=1.0, alpha=0.7, label='train')
    if history['val_loss']:
        ax3.plot(history['eval_iters'], history['val_loss'],
                 color='#ef5350', linewidth=1.5, marker='o', markersize=3, label='val')
    ax3.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=8)

    ax4 = fig.add_subplot(gs[1, 0])
    style_ax(ax4, 'Learning Rate', 'Iteration', 'LR')
    if history['lr']:
        ax4.plot(history['loss_iters'], history['lr'], color='#66bb6a', linewidth=1.2)
        ax4.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))

    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5, 'HRM Reasoning Steps (Train)', 'Iteration', 'Avg Steps')
    if history['train_hrm_steps']:
        ax5.plot(history['loss_iters'], history['train_hrm_steps'],
                 color='#ffa726', linewidth=1.2, alpha=0.9)

    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6, 'HRM Reasoning Steps (Val)', 'Iteration', 'Avg Steps')
    if history['val_hrm_steps']:
        ax6.plot(history['eval_iters'], history['val_hrm_steps'],
                 color='#ab47bc', linewidth=1.5, marker='o', markersize=3)

    plot_path = os.path.join(out_dir, 'training_dashboard.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    args = get_args()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, seed_offset = setup_distributed()

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)

    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if args.dataset_names is None:
        if os.path.exists(os.path.join(args.data_dir, 'train.bin')):
            dataset_names = None
            if master_process:
                print(f"Using single dataset at {args.data_dir}")
        else:
            dataset_names = [
                item for item in os.listdir(args.data_dir)
                if os.path.isdir(os.path.join(args.data_dir, item))
                and os.path.exists(os.path.join(args.data_dir, item, 'train.bin'))
            ]
            if not dataset_names:
                raise ValueError(f"No datasets found in {args.data_dir}")
            if master_process:
                print(f"Found {len(dataset_names)} datasets: {dataset_names}")
    else:
        dataset_names = args.dataset_names
        if master_process:
            print(f"Using specified datasets: {dataset_names}")

    if master_process:
        print("\nInitializing CosmicFish-HRM...")

    iter_num = 0
    best_val_loss = float('inf')

    if args.init_from == 'scratch':
        config = HRMCosmicFishConfig()
        model = HRMCosmicFish(config)
    elif args.init_from == 'resume':
        assert args.resume_path, "--resume_path must be provided when --init_from=resume"
        if master_process:
            print(f"Resuming from: {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=device)
        config = checkpoint['config']
        model = HRMCosmicFish(config)
        state_dict = {
            k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v
            for k, v in checkpoint['model'].items()
        }
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    args.block_size_eval = config.block_size
    model.to(device)

    if master_process:
        print(f"  Architecture: {config.n_input_layers} input + HRM({config.hrm_H_layers}H/{config.hrm_L_layers}L) + {config.n_output_layers} output layers")
        print(f"  Embed dim: {config.n_embd} | Heads: {config.n_head} (KV: {config.n_kv_head}) | Block: {config.block_size}")
        print(f"  Max HRM steps: {config.hrm_max_steps}")

    if args.compile and hasattr(torch, 'compile'):
        if master_process:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
    )

    if args.init_from == 'resume' and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    run = None
    if args.wandb_log and master_process:
        run_name = args.wandb_run_name or f"CosmicFish-HRM-{config.n_embd}d-{time.strftime('%Y%m%d-%H%M%S')}"
        try:
            run = wandb.init(
                project=args.wandb_project,
                entity=WANDB_ENTITY,
                name=run_name,
                config={
                    "model/vocab_size": config.vocab_size,
                    "model/n_embd": config.n_embd,
                    "model/block_size": config.block_size,
                    "model/n_input_layers": config.n_input_layers,
                    "model/n_output_layers": config.n_output_layers,
                    "model/n_head": config.n_head,
                    "model/n_kv_head": config.n_kv_head,
                    "model/hrm_H_layers": config.hrm_H_layers,
                    "model/hrm_L_layers": config.hrm_L_layers,
                    "model/hrm_H_cycles": config.hrm_H_cycles,
                    "model/hrm_L_cycles": config.hrm_L_cycles,
                    "model/hrm_max_steps": config.hrm_max_steps,
                    "model/hrm_exploration_prob": config.hrm_exploration_prob,
                    "model/use_rotary": config.use_rotary,
                    "model/use_gqa": config.use_gqa,
                    "model/use_swiglu": config.use_swiglu,
                    "model/total_params_M": raw_model.get_num_params() / 1e6,
                    "train/batch_size": args.batch_size,
                    "train/gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "train/effective_batch_size": args.batch_size * args.gradient_accumulation_steps * ddp_world_size,
                    "train/max_iters": args.max_iters,
                    "train/learning_rate": args.learning_rate,
                    "train/min_lr": args.min_lr,
                    "train/warmup_iters": args.warmup_iters,
                    "train/weight_decay": args.weight_decay,
                    "train/grad_clip": args.grad_clip,
                    "train/dtype": args.dtype,
                    "train/init_from": args.init_from,
                }
            )
            wandb.define_metric("iter")
            wandb.define_metric("train/*", step_metric="iter")
            wandb.define_metric("val/*", step_metric="iter")
            wandb.define_metric("system/*", step_metric="iter")
            print(f"W&B run initialized: {run.name} ({run.url})")
        except Exception as e:
            print(f"[ERROR] wandb.init failed: {e}")
            sys.exit(1)

    history = {
        'iters': [],
        'loss_iters': [],
        'eval_iters': [],
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'train_hrm_steps': [],
        'val_hrm_steps': [],
        'grad_norm': [],
    }

    if master_process:
        eff_bs = args.batch_size * args.gradient_accumulation_steps * ddp_world_size
        print(f"\nStarting training from iteration {iter_num}")
        print(f"Effective batch size: {eff_bs}")
        print("=" * 80)

    start_time = time.time()

    while iter_num < args.max_iters:
        lr = get_lr(iter_num, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % args.eval_interval == 0 and master_process:
            eval_results = estimate_loss(model, args.data_dir, dataset_names, args, device, ctx)
            train_loss_eval = eval_results['train_loss']
            val_loss_eval   = eval_results['val_loss']
            train_hrm       = eval_results['train_hrm_steps']
            val_hrm         = eval_results['val_hrm_steps']

            if iter_num > 0:
                print()
            print(
                f"[Eval @ {iter_num}] "
                f"train loss: {train_loss_eval:.4f} | val loss: {val_loss_eval:.4f} | "
                f"train HRM steps: {train_hrm:.2f} | val HRM steps: {val_hrm:.2f}"
            )

            history['eval_iters'].append(iter_num)
            history['val_loss'].append(val_loss_eval)
            history['val_hrm_steps'].append(val_hrm)

            if run:
                save_training_plot(history, args.out_dir)
                plot_path = os.path.join(args.out_dir, 'training_dashboard.png')
                log_dict = {
                    "iter": iter_num,
                    "val/loss": val_loss_eval,
                    "val/hrm_steps": val_hrm,
                    "train/loss_eval": train_loss_eval,
                    "train/hrm_steps_eval": train_hrm,
                }
                if os.path.exists(plot_path):
                    log_dict["charts/dashboard"] = wandb.Image(plot_path, caption=f"Step {iter_num}")
                wandb.log(log_dict)

            if val_loss_eval < best_val_loss:
                best_val_loss = val_loss_eval
                if iter_num > 0:
                    ckpt = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'args': vars(args),
                    }
                    torch.save(ckpt, os.path.join(args.out_dir, 'best_model.pt'))
                    print(f"  → Saved best model (val_loss={best_val_loss:.4f})")
                    if run:
                        wandb.log({"iter": iter_num, "val/best_loss": best_val_loss})

        if iter_num % args.save_interval == 0 and iter_num > 0 and master_process:
            ckpt = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(args.out_dir, f'ckpt_{iter_num:06d}.pt'))
            print(f"\n  → Checkpoint saved at iter {iter_num}")

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(args.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)

            if dataset_names:
                dataset_name = np.random.choice(dataset_names)
                data_path = os.path.join(args.data_dir, dataset_name, 'train.bin')
            else:
                data_path = os.path.join(args.data_dir, 'train.bin')

            X, Y = get_batch(data_path, config.block_size, args.batch_size, device)

            with ctx:
                logits, loss, steps_taken, q_logits = model(X, Y)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

        grad_norm = None
        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()

        scaler.step(optimizer)
        scaler.update()

        if master_process and (iter_num % args.log_interval == 0):
            lossf   = loss.item() * args.gradient_accumulation_steps
            avg_hrm = steps_taken.float().mean().item()
            elapsed = time.time() - start_time
            tokens_per_sec = (
                iter_num * args.batch_size * config.block_size * args.gradient_accumulation_steps
            ) / max(elapsed, 1)

            history['iters'].append(iter_num)
            history['loss_iters'].append(iter_num)
            history['train_loss'].append(lossf)
            history['lr'].append(lr)
            history['train_hrm_steps'].append(avg_hrm)
            if grad_norm is not None:
                history['grad_norm'].append(grad_norm)

            gn_str = f"{grad_norm:.3f}" if grad_norm is not None else "n/a"
            pct    = (iter_num / args.max_iters) * 100
            print(
                f"\rIter {iter_num}/{args.max_iters} ({pct:.1f}%) | "
                f"Loss: {lossf:.4f} | LR: {lr:.2e} | "
                f"HRM: {avg_hrm:.1f} steps | "
                f"GradNorm: {gn_str} | "
                f"Tok/s: {tokens_per_sec:.0f}",
                end='', flush=True
            )

            if run:
                log_dict = {
                    "iter": iter_num,
                    "train/loss": lossf,
                    "train/hrm_steps": avg_hrm,
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/elapsed_hours": elapsed / 3600,
                }
                if grad_norm is not None:
                    log_dict["train/grad_norm"] = grad_norm
                if device_type == 'cuda':
                    log_dict["system/gpu_memory_allocated_GB"] = torch.cuda.memory_allocated() / 1e9
                    log_dict["system/gpu_memory_reserved_GB"]  = torch.cuda.memory_reserved() / 1e9
                wandb.log(log_dict)

        iter_num += 1

    if master_process:
        print("\n\nTraining complete.")
        ckpt = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'args': vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, 'final_model.pt'))
        print(f"Final model saved to: {args.out_dir}/final_model.pt")

        save_training_plot(history, args.out_dir)
        plot_path = os.path.join(args.out_dir, 'training_dashboard.png')
        print(f"Training dashboard saved to: {plot_path}")

        if run:
            if os.path.exists(plot_path):
                wandb.log({"charts/final_dashboard": wandb.Image(plot_path, caption="Final")})
            wandb.finish()

    if ddp:
        dist.destroy_process_group()


import torch.nn as nn

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    decay, no_decay = set(), set()
    whitelist = (nn.Linear,)
    blacklist = (nn.LayerNorm, nn.Embedding)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)    if pn in param_dict], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay) if pn in param_dict], "weight_decay": 0.0},
    ]

    use_fused = (device_type == 'cuda') and ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames)
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas,
        **(dict(fused=True) if use_fused else {})
    )
    return optimizer


HRMCosmicFish.configure_optimizers = configure_optimizers


if __name__ == '__main__':
    main()