import os
import sys
import time
import math
import argparse
import logging
import pickle
import numpy as np
import torch
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import HRMCosmicFish, HRMCosmicFishConfig
from torch.serialization import add_safe_globals

add_safe_globals([HRMCosmicFishConfig])

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def clean_state_dict_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        k = key.removeprefix('_orig_mod.').removeprefix('module.')
        cleaned[k] = value
    return cleaned


def diagnose_state_dict_mismatch(checkpoint_state_dict, model_state_dict):
    checkpoint_keys = set(checkpoint_state_dict.keys())
    model_keys      = set(model_state_dict.keys())
    missing_keys    = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys

    logger.info(f"Checkpoint: {len(checkpoint_keys)} keys | Model expects: {len(model_keys)} keys")

    if missing_keys:
        logger.warning(f"Missing {len(missing_keys)} keys:")
        for key in sorted(list(missing_keys)[:5]):
            logger.warning(f"  - {key}")
        if len(missing_keys) > 5:
            logger.warning(f"  ... and {len(missing_keys) - 5} more")

    if unexpected_keys:
        logger.warning(f"Unexpected {len(unexpected_keys)} keys:")
        for key in sorted(list(unexpected_keys)[:5]):
            logger.warning(f"  + {key}")
        if len(unexpected_keys) > 5:
            logger.warning(f"  ... and {len(unexpected_keys) - 5} more")

    for prefix in ['_orig_mod.', 'module.', '_forward_module.']:
        if any(key.startswith(prefix) for key in checkpoint_keys):
            logger.info(f"Found '{prefix}' prefix in checkpoint keys")


def get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr, decay_type='cosine'):
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    if iter_num > lr_decay_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) if decay_type == 'cosine' else 1.0 - decay_ratio
    return min_lr + coeff * (learning_rate - min_lr)


def setup_model_and_optimizer(args, device, iter_num=0):
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model checkpoint not found at {args.model_path}")

    logger.info(f"Loading model from {args.model_path}")

    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    except Exception as e:
        logger.warning(f"weights_only=True failed: {e}, falling back")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError("No configuration found in checkpoint")

    model = HRMCosmicFish(config)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        raise ValueError("Could not find model weights in checkpoint")

    cleaned_state_dict = clean_state_dict_keys(state_dict)

    try:
        model.load_state_dict(cleaned_state_dict)
        logger.info("Model weights loaded successfully")
    except RuntimeError as e:
        logger.error(f"Failed to load state dict: {e}")
        diagnose_state_dict_mismatch(cleaned_state_dict, model.state_dict())
        logger.warning("Attempting strict=False...")
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
        if len(missing_keys) > len(model.state_dict()) * 0.1:
            raise RuntimeError("Checkpoint incompatible — too many missing parameters")

    model.to(device)
    logger.info(f"Model loaded: {model.get_num_params() / 1e6:.1f}M parameters")
    logger.info(f"  Input blocks: {config.n_input_layers} | HRM: H={config.hrm_H_layers} L={config.hrm_L_layers} (max {config.hrm_max_steps} steps) | Output blocks: {config.n_output_layers}")

    decay_params    = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(x in name for x in ['bias', 'ln', 'norm', 'wte', 'wpe']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {'params': decay_params,    'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    use_fused = (device == 'cuda') and ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames)
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        **(dict(fused=True) if use_fused else {})
    )

    logger.info(f"Optimizer: {len(decay_params)} decay params, {len(no_decay_params)} no-decay params")

    if args.resume and iter_num > 0 and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr = get_lr(iter_num, args.warmup_iters, args.learning_rate, args.lr_decay_iters, args.min_lr, args.decay_type)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return model, optimizer, config


def get_mixed_batch(identity_data_train, identity_data_val, conv_data_train, conv_data_val,
                    block_size, batch_size, device, split='train', identity_ratio=0.3):
    identity_data = identity_data_train if split == 'train' else identity_data_val
    conv_data     = conv_data_train     if split == 'train' else conv_data_val

    identity_count = int(batch_size * identity_ratio)
    conv_count     = batch_size - identity_count

    if identity_count > 0 and len(identity_data) > block_size:
        identity_ix = torch.randint(len(identity_data) - block_size, (identity_count,))
        identity_x  = torch.stack([torch.from_numpy((identity_data[i:i + block_size]).astype(np.int64)) for i in identity_ix])
        identity_y  = torch.stack([torch.from_numpy((identity_data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in identity_ix])
    else:
        identity_x = torch.empty((0, block_size), dtype=torch.long)
        identity_y = torch.empty((0, block_size), dtype=torch.long)

    if conv_count > 0 and len(conv_data) > block_size:
        conv_ix = torch.randint(len(conv_data) - block_size, (conv_count,))
        conv_x  = torch.stack([torch.from_numpy((conv_data[i:i + block_size]).astype(np.int64)) for i in conv_ix])
        conv_y  = torch.stack([torch.from_numpy((conv_data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in conv_ix])
    else:
        conv_x = torch.empty((0, block_size), dtype=torch.long)
        conv_y = torch.empty((0, block_size), dtype=torch.long)

    if identity_x.size(0) > 0 and conv_x.size(0) > 0:
        x = torch.cat([identity_x, conv_x], dim=0)
        y = torch.cat([identity_y, conv_y], dim=0)
    elif identity_x.size(0) > 0:
        x, y = identity_x, identity_y
    else:
        x, y = conv_x, conv_y

    perm = torch.randperm(x.size(0))
    x, y = x[perm].to(device), y[perm].to(device)
    return x, y


@torch.no_grad()
def evaluate_mixed(model, identity_data_train, identity_data_val, conv_data_train, conv_data_val,
                   block_size, batch_size, device, eval_iters=20, ctx=nullcontext()):
    model.eval()

    identity_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_mixed_batch(identity_data_train, identity_data_val, conv_data_train, conv_data_val,
                               block_size, batch_size, device, 'val', identity_ratio=1.0)
        with ctx:
            _, loss, _, _ = model(X, Y)
        identity_losses[k] = loss.item()

    conv_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_mixed_batch(identity_data_train, identity_data_val, conv_data_train, conv_data_val,
                               block_size, batch_size, device, 'val', identity_ratio=0.0)
        with ctx:
            _, loss, _, _ = model(X, Y)
        conv_losses[k] = loss.item()

    mixed_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_mixed_batch(identity_data_train, identity_data_val, conv_data_train, conv_data_val,
                               block_size, batch_size, device, 'val', identity_ratio=0.3)
        with ctx:
            _, loss, _, _ = model(X, Y)
        mixed_losses[k] = loss.item()

    model.train()
    return identity_losses.mean().item(), conv_losses.mean().item(), mixed_losses.mean().item()


def save_checkpoint(model, optimizer, iter_num, best_val_loss, args, config, is_best=False):
    os.makedirs(args.output_dir, exist_ok=True)
    filename = 'best_calibrated.pt' if is_best else f'calibrated_{iter_num:06d}.pt'
    filepath = os.path.join(args.output_dir, filename)

    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

    checkpoint = {
        'model':                  model_to_save.state_dict(),
        'optimizer':              optimizer.state_dict(),
        'iter_num':               iter_num,
        'best_val_loss':          best_val_loss,
        'config':                 config,
        'calibration_completed':  True,
        'pytorch_version':        torch.__version__,
        'training_stage':         'identity_calibration'
    }

    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")

    if not is_best:
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest_calibrated.pt'))


def main():
    parser = argparse.ArgumentParser(description="Identity calibration for HRM-CosmicFish")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--identity_dir", type=str, default="data/identity")
    parser.add_argument("--conv_dir", type=str, default="data/alpaca_gpt4_cleaned_pure")
    parser.add_argument("--output_dir", type=str, default="out/calibrated")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_iters", type=int, default=100000)

    parser.add_argument("--warmup_iters", type=int, default=50)
    parser.add_argument("--lr_decay_iters", type=int, default=100000)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--decay_type", type=str, default="cosine", choices=["cosine", "linear"])

    parser.add_argument("--identity_ratio", type=float, default=0.4)
    parser.add_argument("--early_stop_threshold", type=float, default=0.15)

    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str,
                        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    logger.info(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")

    for path, name in [
        (os.path.join(args.identity_dir, 'train.bin'), "Identity train"),
        (os.path.join(args.identity_dir, 'val.bin'),   "Identity val"),
        (os.path.join(args.conv_dir, 'train.bin'),     "Conversational train"),
        (os.path.join(args.conv_dir, 'val.bin'),       "Conversational val"),
    ]:
        if not os.path.exists(path):
            logger.error(f"{name} not found: {path}")
            sys.exit(1)

    logger.info("Loading datasets...")
    identity_train_data = np.memmap(os.path.join(args.identity_dir, 'train.bin'), dtype=np.uint16, mode='r')
    identity_val_data   = np.memmap(os.path.join(args.identity_dir, 'val.bin'),   dtype=np.uint16, mode='r')
    conv_train_data     = np.memmap(os.path.join(args.conv_dir, 'train.bin'),     dtype=np.uint16, mode='r')
    conv_val_data       = np.memmap(os.path.join(args.conv_dir, 'val.bin'),       dtype=np.uint16, mode='r')

    logger.info(f"Identity: {len(identity_train_data):,} train / {len(identity_val_data):,} val tokens")
    logger.info(f"Conv:     {len(conv_train_data):,} train / {len(conv_val_data):,} val tokens")

    device = args.device

    if device == 'cuda':
        amp_dtype = torch.bfloat16 if args.dtype == 'bfloat16' and torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype)
        logger.info(f"Mixed precision: {amp_dtype}")
    else:
        ctx = nullcontext()

    iter_num       = 0
    best_val_loss  = float('inf')
    initial_conv_loss = None

    if args.resume:
        latest_path = os.path.join(args.output_dir, 'latest_calibrated.pt')
        if os.path.exists(latest_path):
            try:
                ckpt = torch.load(latest_path, map_location=device, weights_only=True)
            except:
                ckpt = torch.load(latest_path, map_location=device, weights_only=False)
            iter_num      = ckpt.get('iter_num', 0)
            best_val_loss = ckpt.get('best_val_loss', float('inf'))
            logger.info(f"Resuming from iteration {iter_num}, best val loss {best_val_loss:.4f}")

    model, optimizer, config = setup_model_and_optimizer(args, device, iter_num)
    block_size = config.block_size

    logger.info("Evaluating baseline...")
    identity_loss, conv_loss, mixed_loss = evaluate_mixed(
        model, identity_train_data, identity_val_data, conv_train_data, conv_val_data,
        block_size, args.batch_size, device, args.eval_iters, ctx
    )
    initial_conv_loss = conv_loss
    logger.info(f"Baseline — Identity: {identity_loss:.4f} | Conv: {conv_loss:.4f} | Mixed: {mixed_loss:.4f}")

    logger.info(f"\nCalibration: iters {iter_num}→{args.max_iters} | batch {args.batch_size} | identity ratio {args.identity_ratio} | lr {args.learning_rate}\n")

    t0 = time.time()
    while iter_num < args.max_iters:
        lr = get_lr(iter_num, args.warmup_iters, args.learning_rate, args.lr_decay_iters, args.min_lr, args.decay_type)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for micro_step in range(args.gradient_accumulation_steps):
            X, Y = get_mixed_batch(identity_train_data, identity_val_data, conv_train_data, conv_val_data,
                                   block_size, args.batch_size, device, 'train', args.identity_ratio)
            with ctx:
                _, loss, _, _ = model(X, Y)
                loss = loss / args.gradient_accumulation_steps
            total_loss += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if iter_num % args.log_interval == 0:
            t1 = time.time()
            tokens_per_sec = args.batch_size * block_size * args.gradient_accumulation_steps / (t1 - t0)
            logger.info(f"Iter {iter_num}: loss {total_loss:.4f} | lr {lr:.2e} | {tokens_per_sec:.1f} tok/s")
            t0 = t1

        if iter_num % args.eval_interval == 0:
            identity_loss, conv_loss, mixed_loss = evaluate_mixed(
                model, identity_train_data, identity_val_data, conv_train_data, conv_val_data,
                block_size, args.batch_size, device, args.eval_iters, ctx
            )
            logger.info(f"Iter {iter_num}: Identity {identity_loss:.4f} | Conv {conv_loss:.4f} | Mixed {mixed_loss:.4f}")

            if initial_conv_loss is not None and conv_loss > initial_conv_loss + args.early_stop_threshold:
                logger.warning(f"Conv loss increased by {conv_loss - initial_conv_loss:.4f}, stopping early!")
                break

            if mixed_loss < best_val_loss:
                best_val_loss = mixed_loss
                save_checkpoint(model, optimizer, iter_num, best_val_loss, args, config, is_best=True)
                logger.info(f"New best mixed val loss: {best_val_loss:.4f}")

        if iter_num % args.save_interval == 0 or iter_num == args.max_iters - 1:
            save_checkpoint(model, optimizer, iter_num, best_val_loss, args, config)

        iter_num += 1

    logger.info(f"\nCalibration complete | Best mixed val loss: {best_val_loss:.4f} | Saved to: {args.output_dir}/best_calibrated.pt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during calibration: {str(e)}", exc_info=True)
        sys.exit(1)