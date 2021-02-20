import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image

import init_helper
import metrics
from dataloader import dataset_utils
from dataloader.dataset import TestDataset
from modules.ssm import SSM

logger = logging.getLogger()


def evaluate(model, loader, log_step, save_dir=None):
    eval_ious = []
    frame_times = []
    model = model.eval()

    with torch.no_grad():
        for batch_idx, (ref_img, ref_mask, cur_img, cur_mask, prev_mask, crop_box) in enumerate(loader):
            frame_info = loader.dataset.sequence[batch_idx]

            crop_box = crop_box[0]
            x1, y1, x2, y2 = crop_box
            box_w = x2 - x1
            box_h = y2 - y1

            ref_img = ref_img.cuda()
            cur_img = cur_img.cuda()
            ref_mask = ref_mask.cuda()
            prev_mask = prev_mask.cuda()

            start = time.time()

            pred_mask, _ = model(ref_img, ref_mask, cur_img, prev_mask)
            pred_mask = torch.softmax(pred_mask, dim=1)
            pred_mask = F.interpolate(pred_mask, (box_h, box_w), mode='bilinear', align_corners=False)

            org_cur_mask = loader.dataset.get_mask(batch_idx)
            org_h, org_w = org_cur_mask.shape

            pred_mask = pred_mask.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            pred_mask = dataset_utils.crop_image_back(pred_mask, crop_box, org_h, org_w)

            frame_time = time.time() - start
            frame_times.append(frame_time)

            if save_dir is not None:
                video_dir = Path(save_dir) / frame_info['video_name'] / str(frame_info['object_index'])
                video_dir.mkdir(parents=True, exist_ok=True)
                save_stem = Path(frame_info['image_path']).stem
                if frame_info['frame_index'] == 0:
                    # First frame of the video, take the ground truth
                    save_mask = loader.dataset.get_mask(batch_idx)
                else:
                    # Middle of the video, take the predicted frame
                    save_mask = pred_mask[:, :, 1]

                save_mask = Image.fromarray((save_mask * 255).astype(np.uint8))
                save_mask.save(video_dir / f'{save_stem}.png')

            if frame_info['frame_index'] == 0:
                pred_mask = loader.dataset.get_mask(batch_idx)
            else:
                pred_mask = np.argmax(pred_mask, axis=-1).astype(np.float32)

            loader.dataset.prev_mask = pred_mask

            eval_iou = metrics.iou(org_cur_mask, pred_mask)
            eval_ious.append(eval_iou)

            if (batch_idx + 1) % log_step == 0:
                logger.info(f'step {batch_idx}: iou: {eval_iou:.4f}, time: {frame_time:.4f}')

    eval_iou = np.mean(eval_ious)
    frame_time = np.mean(frame_times)

    return eval_iou, frame_time


def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(None)
    init_helper.set_random_seed(args.seed)

    logger.info(f'Evaluating model {args.resume}')

    model = SSM(encoder=args.encoder, matching=args.matching, keep_topk=args.keep_topk).cuda()
    checkpoint = torch.load(args.resume, map_location=lambda storage, location: storage)
    model_dict = checkpoint['model_dict']
    model.load_state_dict(model_dict)
    model = model.eval()

    logger.info(f'args: {vars(args)}')

    with open(args.split) as f:
        seq = json.load(f)

    dataset = TestDataset(seq, args.input_height, args.input_width, args.fix_margin)

    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    iou, frame_time = evaluate(model, loader, args.log_step, save_dir=args.save_dir)
    logger.info(f'iou: {iou:.4f}, frame_time: {frame_time:.4f}')


if __name__ == '__main__':
    main()
