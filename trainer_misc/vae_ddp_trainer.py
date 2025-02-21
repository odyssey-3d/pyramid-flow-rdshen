import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torchvision
import wandb

from .utils import (
    MetricLogger,
    SmoothedValue,
)


def train_one_epoch(
    model: torch.nn.Module, 
    model_dtype: str,
    data_loader: Iterable, 
    optimizer: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer,
    device: torch.device, 
    epoch: int, 
    loss_scaler, 
    loss_scaler_disc,
    clip_grad: float = 0,
    loggers=None, 
    lr_scheduler=None, 
    start_steps=None,
    lr_schedule_values=None,
    lr_schedule_values_disc=None,
    args=None,
    print_freq=20,
    vis_freq=20,
    iters_per_epoch=2000,
):
    # The trainer for causal video vae

    model.train()
    metric_logger = MetricLogger(delimiter="  ")

    if optimizer is not None:
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    if optimizer_disc is not None:
        metric_logger.add_meter('disc_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('disc_min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)

    if model_dtype == 'bf16':
        _dtype = torch.bfloat16
    else:
        _dtype = torch.float16

    print("Start training epoch {}, {} iters per inner epoch.".format(epoch, iters_per_epoch))

    # Helper function to safely send wandb alerts
    def send_wandb_alert(title, text, level):
        if loggers is not None and 'wandb' in loggers:
            run = loggers['wandb']
            context = f"Epoch: {epoch}, Step: {start_steps + step}\n"
            context += f"Run: {run.name} ({run.get_url()})\n\n"
            full_text = context + text
            
            run.alert(
                title=title,
                text=full_text,
                level=level,
                wait_duration=300
            )
            # Also print to stdout with context
            print(f"[Alert] {title} at epoch {epoch}, step {start_steps + step}:\n{text}")

    try:
        for step in metric_logger.log_every(range(iters_per_epoch), print_freq, header):
            if step >= iters_per_epoch:
                break
            
            it = start_steps + step  # global training iteration
            if lr_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)

            if optimizer_disc is not None:
                for i, param_group in enumerate(optimizer_disc.param_groups):
                    if lr_schedule_values_disc is not None:
                        param_group["lr"] = lr_schedule_values_disc[it] * param_group.get("lr_scale", 1.0)

            samples = next(data_loader)
        
            samples['video'] = samples['video'].to(device, non_blocking=True)

            # Only return reconstructions if we're going to visualize this step
            should_visualize = loggers is not None and it % vis_freq == 0
            
            with torch.cuda.amp.autocast(enabled=True, dtype=_dtype):
                outputs = model(samples['video'], args.global_step, identifier=samples['identifier'], 
                              return_recons=should_visualize)
                
                if should_visualize:
                    rec_loss, gan_loss, log_loss, reconstruct = outputs
                else:
                    rec_loss, gan_loss, log_loss = outputs

            ###################################################################################################
            # The update of rec_loss
            if rec_loss is not None:
                loss_value = rec_loss.item()

                if not math.isfinite(loss_value):
                    send_wandb_alert(
                        "NaN Loss Detected",
                        f"Loss value is {loss_value}. Stopping training.",
                        wandb.AlertLevel.ERROR
                    )
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                optimizer.zero_grad()
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(rec_loss, optimizer, clip_grad=clip_grad,
                                        parameters=model.module.vae.parameters(), create_graph=is_second_order)
                
                if "scale" in loss_scaler.state_dict():
                    loss_scale_value = loss_scaler.state_dict()["scale"]
                else:
                    loss_scale_value = 1

                metric_logger.update(vae_loss=loss_value)
                metric_logger.update(loss_scale=loss_scale_value)

                # Add check for loss spikes
                if loss_value >= 2.0:
                    send_wandb_alert(
                        "Loss Spike Detected",
                        f"Extreme high loss detected: {loss_value}.",
                        wandb.AlertLevel.WARN
                    )

            ###################################################################################################

            # The update of gan_loss
            if gan_loss is not None:
                gan_loss_value = gan_loss.item()
                
                if not math.isfinite(gan_loss_value):
                    send_wandb_alert(
                        "NaN GAN Loss Detected",
                        f"The gan discriminator Loss is {gan_loss_value}. Stopping training.",
                        wandb.AlertLevel.ERROR
                    )
                    print("The gan discriminator Loss is {}, stopping training".format(gan_loss_value), force=True)
                    sys.exit(1)

                optimizer_disc.zero_grad()
                is_second_order = hasattr(optimizer_disc, 'is_second_order') and optimizer_disc.is_second_order
                disc_grad_norm = loss_scaler_disc(gan_loss, optimizer_disc, clip_grad=clip_grad,
                                        parameters=model.module.loss.discriminator.parameters(), create_graph=is_second_order)

                if "scale" in loss_scaler_disc.state_dict():
                    disc_loss_scale_value = loss_scaler_disc.state_dict()["scale"]
                else:
                    disc_loss_scale_value = 1

                metric_logger.update(disc_loss=gan_loss_value)
                metric_logger.update(disc_loss_scale=disc_loss_scale_value)
                metric_logger.update(disc_grad_norm=disc_grad_norm)

                min_lr = 10.
                max_lr = 0.
                for group in optimizer_disc.param_groups:
                    min_lr = min(min_lr, group["lr"])
                    max_lr = max(max_lr, group["lr"])

                metric_logger.update(disc_lr=max_lr)
                metric_logger.update(disc_min_lr=min_lr)

                # Add check for GAN loss spikes
                if gan_loss_value >= 2.0:
                    send_wandb_alert(
                        "GAN Loss Spike Detected",
                        f"Extreme high GAN loss detected: {gan_loss_value}.",
                        wandb.AlertLevel.WARN
                    )

            torch.cuda.synchronize()
            new_log_loss = {k.split('/')[-1]:v for k, v in log_loss.items() if k not in ['total_loss']}
            metric_logger.update(**new_log_loss)

            if rec_loss is not None:
                min_lr = 10.
                max_lr = 0.
                for group in optimizer.param_groups:
                    min_lr = min(min_lr, group["lr"])
                    max_lr = max(max_lr, group["lr"])

                metric_logger.update(lr=max_lr)
                metric_logger.update(min_lr=min_lr)
                weight_decay_value = None
                for group in optimizer.param_groups:
                    if group["weight_decay"] > 0:
                        weight_decay_value = group["weight_decay"]
                metric_logger.update(weight_decay=weight_decay_value)
                metric_logger.update(grad_norm=grad_norm)

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step)

            args.global_step = args.global_step + 1

            if loggers is not None and it % print_freq == 0:
                for name, value in metric_logger.meters.items():
                    if 'tensorboard' in loggers:
                        loggers['tensorboard'].add_scalar(f'train/{name}', value.global_avg, args.global_step)
                    if 'wandb' in loggers:
                        wandb.log({f'train/{name}': value.global_avg}, step=args.global_step)

                if should_visualize:
                    with torch.no_grad():
                        input_images = samples['video'].detach().cpu().float()
                        recon_images = reconstruct.detach().cpu().float()
                        
                        def create_comparison_grid(inputs, recons, max_frames=16):
                            """Create a grid comparing input and reconstruction frames."""
                            B, _, T, _, _ = inputs.shape
                            
                            # First, create a grid comparing first frames side by side
                            first_frames = []
                            for b in range(B):
                                # Get first frame from input and recon
                                first_input = inputs[b:b+1, :, 0]  # (1, C, H, W)
                                first_recon = recons[b:b+1, :, 0]  # (1, C, H, W)
                                # Stack input and recon horizontally
                                first_pair = torch.cat([first_input, first_recon], dim=0)
                                first_frames.append(first_pair)
                            
                            # Combine all first frame pairs
                            first_frames = torch.cat(first_frames, dim=0)  # (2B, C, H, W)
                            first_frame_grid = torchvision.utils.make_grid(first_frames, nrow=2, padding=2, normalize=True)
                            
                            sequence_grid_combined = None
                            if T > 1:  # If we have a video sequence
                                # For each batch item, create a grid of subsequent frames (up to max_frames)
                                sequence_grids = []
                                for b in range(B):
                                    # Get subsequent frames (skip first frame)
                                    seq_input = inputs[b:b+1, :, 1:max_frames+1]  # (1, C, T-1, H, W)
                                    seq_recon = recons[b:b+1, :, 1:max_frames+1]  # (1, C, T-1, H, W)
                                    
                                    # Reshape and interleave input and reconstruction frames
                                    seq_input = seq_input.squeeze(0).transpose(0, 1)  # (T-1, C, H, W)
                                    seq_recon = seq_recon.squeeze(0).transpose(0, 1)  # (T-1, C, H, W)
                                    
                                    # Interleave input and reconstruction frames
                                    num_frames = seq_input.size(0)
                                    sequence_pairs = []
                                    for t in range(num_frames):
                                        frame_pair = torch.stack([seq_input[t], seq_recon[t]], dim=0)
                                        sequence_pairs.append(frame_pair)
                                    sequence_pairs = torch.cat(sequence_pairs, dim=0)  # (2T, C, H, W)
                                    
                                    # Create grid with input-reconstruction pairs side by side
                                    sequence_grid = torchvision.utils.make_grid(
                                        sequence_pairs, 
                                        nrow=8,  # 4 pairs per row (8 images total)
                                        padding=2, 
                                        normalize=True
                                    )
                                    sequence_grids.append(sequence_grid)
                                
                                # Combine all sequence grids vertically
                                if sequence_grids:
                                    sequence_grid_combined = torch.cat(sequence_grids, dim=1)
                            
                            return first_frame_grid, sequence_grid_combined
                        
                        if input_images.dim() == 4:
                            input_images = input_images.unsqueeze(2)  # (B, C, 1, H, W)
                            recon_images = recon_images.unsqueeze(2)  # (B, C, 1, H, W)
                        first_frame_grid, sequence_grid_combined = create_comparison_grid(input_images, recon_images)
                        
                        if 'tensorboard' in loggers:
                            loggers['tensorboard'].add_image(
                                'train/reconstructions_comparison', 
                                first_frame_grid, 
                                args.global_step,
                            )
                            if sequence_grid_combined is not None:
                                loggers['tensorboard'].add_image(
                                    'train/sequence_comparison',
                                    sequence_grid_combined,
                                    args.global_step,   
                                )
                        if 'wandb' in loggers:
                            wandb.log({
                                'train/reconstructions_comparison': wandb.Image(first_frame_grid)
                            }, step=args.global_step)
                            if sequence_grid_combined is not None:
                                wandb.log({
                                    'train/sequence_comparison': wandb.Image(sequence_grid_combined)
                                }, step=args.global_step)

    except Exception as e:
        import traceback
        send_wandb_alert(
            "Training Error",
            f"Error during training:\n{traceback.format_exc()}",
            wandb.AlertLevel.ERROR
        )
        raise e

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
