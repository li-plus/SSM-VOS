import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import init_helper
import metrics
from dataloader.dataset import TrainDataset
from modules.ssm import SSM

logger = logging.getLogger()


def main():
    args = init_helper.get_arguments()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = model_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = model_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    init_helper.init_logger(str(log_dir / 'train.log'))
    init_helper.set_random_seed(args.seed)

    logger.info(f'args: {vars(args)}')

    with open(model_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Build model
    global_step = 0
    model = SSM(encoder=args.encoder, matching=args.matching, keep_topk=args.keep_topk).cuda()

    if args.resume is not None:
        # Load checkpoint
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_dict'])
        if args.resume_step:
            global_step = checkpoint['global_step'] + 1

    model.train()
    parallel_model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # Build optimizer
    params = [x for x in model.parameters() if x.requires_grad]
    optimizer = Adam(params, lr=args.base_lr, weight_decay=args.weight_decay)

    # LR decay
    lr_manager = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # Load training split
    with open(args.split) as f:
        train_seq = json.load(f)

    # Build training dataset
    train_dataset = TrainDataset(train_seq, args.input_height, args.input_width, args.min_margin,
                                 args.max_margin, args.flip_rate, args.rotation, args.blur_rate)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Start training
    with SummaryWriter(str(model_dir / 'board')) as writer:
        for epoch in range(args.max_epoch):
            logger.info(f'training epoch {epoch}')

            train_losses = []
            train_ious = []

            for batch_index, (ref_img, ref_mask, cur_img, cur_mask, prev_mask, crop_box) in enumerate(train_loader):
                ref_img = ref_img.cuda()
                cur_img = cur_img.cuda()
                ref_mask = ref_mask.cuda()
                cur_mask = cur_mask.cuda()
                prev_mask = prev_mask.cuda()

                pred_mask, struct_mask = parallel_model(ref_img, ref_mask, cur_img, prev_mask)
                pred_mask = pred_mask.permute(0, 2, 3, 1).contiguous().view(-1, 2)
                struct_mask = struct_mask.permute(0, 2, 3, 1).contiguous().view(-1, 2)
                cur_mask = cur_mask.view(-1).type(torch.long)
                loss = criterion(pred_mask, cur_mask) + 0.1 * criterion(struct_mask, cur_mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.data.item())

                cur_mask = cur_mask.cpu().numpy()

                pred_mask = torch.argmax(pred_mask, dim=1)
                pred_mask = pred_mask.cpu().numpy()

                train_iou = metrics.iou(cur_mask, pred_mask)
                train_ious.append(train_iou)

                # Tensorboard
                writer.add_scalar('train/loss', loss.data.item(), global_step=global_step)
                writer.add_scalar('train/iou', train_iou, global_step=global_step)

                if global_step % args.lr_decay_step == 0:
                    for idx, param_group in enumerate(optimizer.param_groups):
                        group_lr = param_group['lr']
                        logger.info(f'group {idx}: learning_rate: {group_lr}')
                        writer.add_scalar(f'lr/group_{idx}', group_lr, global_step=global_step)

                if (global_step + 1) % args.log_step == 0:
                    # Logging
                    logger.info(f'step {global_step} ({epoch}, {batch_index}): '
                                f'loss: {loss.data.item():.4f}, iou: {train_iou:.4f}')

                if (global_step + 1) % args.save_step == 0:
                    # Save model
                    ckpt_path = str(ckpt_dir / f'{global_step}.pt')
                    ckpt = {'model_dict': model.state_dict(), 'global_step': global_step}
                    logger.info(f'step {global_step} ({epoch}, {batch_index}): saving checkpoint to {ckpt_path}')
                    torch.save(ckpt, ckpt_path)

                lr_manager.step()
                global_step += 1

            train_loss = np.mean(train_losses)
            train_iou = np.mean(train_ious)

            logger.info(f'epoch {epoch}: train_loss: {train_loss:.4f}, iou: {train_iou:.4f}')


if __name__ == '__main__':
    main()
