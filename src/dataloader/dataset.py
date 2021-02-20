import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from dataloader import dataset_utils


class BaseDataset(data.Dataset):
    def __init__(self, sequence):
        self.sequence = sequence
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.sequence)

    def get_image(self, index):
        return np.array(Image.open(self.sequence[index]['image_path']))

    def get_mask(self, index):
        frame_info = self.sequence[index]
        mask = np.array(Image.open(frame_info['mask_path']))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask == frame_info['object_index']).astype(np.float32)
        return mask

    def get_image_mask(self, index):
        return self.get_image(index), self.get_mask(index)

    @staticmethod
    def flip(image, mask, bounding_mask):
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
        bounding_mask = np.flip(bounding_mask, axis=1).copy()
        return image, mask, bounding_mask

    @staticmethod
    def rotate(image, mask, bounding_mask, degree):
        image = dataset_utils.rotate_bound(image, degree, cv2.INTER_LINEAR)
        mask = dataset_utils.rotate_bound(mask, degree, cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.float32)
        bounding_mask = dataset_utils.rotate_bound(bounding_mask, degree, cv2.INTER_LINEAR)
        bounding_mask = (bounding_mask > 0.5).astype(np.float32)
        return image, mask, bounding_mask

    @staticmethod
    def crop(image, mask, bounding_mask, margin_left, margin_right, margin_top, margin_bottom):
        img_h, img_w = mask.shape
        crop_box = np.array([0, 0, img_w, img_h], dtype=np.int32)

        if not np.isclose(bounding_mask, 0).all():
            x1, y1, box_w, box_h = cv2.boundingRect(bounding_mask.astype(np.uint8))
            x2 = x1 + box_w
            y2 = y1 + box_h

            x1 -= box_w * margin_left
            x2 += box_w * margin_right
            y1 -= box_h * margin_top
            y2 += box_h * margin_bottom

            if x2 - x1 > 5 and y2 - y1 > 5:
                crop_box = np.array([x1, y1, x2, y2], dtype=np.int32)

                image = dataset_utils.crop_image(image, crop_box)
                mask = dataset_utils.crop_image(mask, crop_box)
                bounding_mask = dataset_utils.crop_image(bounding_mask, crop_box)

        return image, mask, bounding_mask, crop_box

    @staticmethod
    def resize(image, mask, bounding_mask, out_h, out_w):
        image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.float32)
        bounding_mask = cv2.resize(bounding_mask, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        bounding_mask = (bounding_mask > 0.5).astype(np.float32)
        return image, mask, bounding_mask


class TrainDataset(BaseDataset):
    def __init__(self, sequence, img_h, img_w, min_margin, max_margin,
                 flip_rate, degree, blur_rate):
        super().__init__(sequence)
        self.img_h = img_h
        self.img_w = img_w
        self.flip_rate = flip_rate
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.degree = degree
        self.blur_rate = blur_rate

    def get_positive_pair(self, index):
        cur_img, cur_mask = self.get_image_mask(index)

        num_frames = self.sequence[index]['num_frames']
        frame_index = self.sequence[index]['frame_index']
        ref_index = random.randint(index - frame_index, index - frame_index + num_frames - 1)

        ref_img, ref_mask = self.get_image_mask(ref_index)

        return ref_img, ref_mask, cur_img, cur_mask

    def get_prev_mask(self, index):
        if self.sequence[index]['frame_index'] == 0:
            prev_index = index
        else:
            prev_index = index - 1

        _, prev_mask = self.get_image_mask(prev_index)

        return prev_mask

    def process_image_mask(self, image, mask, bounding_mask, out_h, out_w):
        # flip
        if random.random() < self.flip_rate:
            image, mask, bounding_mask = self.flip(image, mask, bounding_mask)

        # rotate
        degree = random.uniform(-self.degree, self.degree)
        image, mask, bounding_mask = self.rotate(image, mask, bounding_mask, degree)

        # crop
        margin_top, margin_bottom, margin_left, margin_right = [random.uniform(self.min_margin, self.max_margin)
                                                                for _ in range(4)]
        image, mask, bounding_mask, crop_box = self.crop(image, mask, bounding_mask, margin_left, margin_right,
                                                         margin_top, margin_bottom)

        # resize
        image, mask, bounding_mask = self.resize(image, mask, bounding_mask, out_h, out_w)

        return image, mask, bounding_mask, crop_box

    def __getitem__(self, index):
        ref_img, ref_mask, cur_img, cur_mask = self.get_positive_pair(index)
        prev_mask = self.get_prev_mask(index)

        ref_img, ref_mask, _, _ = self.process_image_mask(
            ref_img, ref_mask, ref_mask.copy(), self.img_h, self.img_w)
        cur_img, cur_mask, prev_mask, crop_box = self.process_image_mask(
            cur_img, cur_mask, prev_mask, self.img_h, self.img_w)

        # blur previous mask
        if random.random() < self.blur_rate:
            prev_mask = dataset_utils.get_gb_image(prev_mask)

        # transform to tensor
        cur_img = self.transform(cur_img)
        ref_img = self.transform(ref_img)

        cur_mask = torch.tensor(cur_mask, dtype=torch.float32).unsqueeze(0)
        ref_mask = torch.tensor(ref_mask, dtype=torch.float32).unsqueeze(0)
        prev_mask = torch.tensor(prev_mask, dtype=torch.float32).unsqueeze(0)

        return ref_img, ref_mask, cur_img, cur_mask, prev_mask, crop_box


class TestDataset(BaseDataset):

    def __init__(self, sequence, img_h, img_w, margin):
        super().__init__(sequence)
        self.img_h = img_h
        self.img_w = img_w
        self.margin = margin
        self.prev_mask = None

    def process_image_mask(self, image, mask, bounding_mask, out_h, out_w):
        image, mask, bounding_mask, crop_box = self.crop(image, mask, bounding_mask, self.margin, self.margin,
                                                         self.margin, self.margin)
        image, mask, bounding_mask = self.resize(image, mask, bounding_mask, out_h, out_w)

        return image, mask, bounding_mask, crop_box

    def __getitem__(self, index):

        cur_img, cur_mask = self.get_image_mask(index)
        ref_img, ref_mask = self.get_image_mask(index - self.sequence[index]['frame_index'])

        if self.sequence[index]['frame_index'] == 0:
            prev_mask = cur_mask
        else:
            prev_mask = self.prev_mask

        ref_img, ref_mask, _, _ = self.process_image_mask(
            ref_img, ref_mask, ref_mask.copy(), self.img_h, self.img_w)
        cur_img, cur_mask, prev_mask, crop_box = self.process_image_mask(
            cur_img, cur_mask, prev_mask, self.img_h, self.img_w)

        # transform to tensor
        cur_img = self.transform(cur_img)
        ref_img = self.transform(ref_img)

        cur_mask = torch.tensor(cur_mask, dtype=torch.float32).unsqueeze(0)
        ref_mask = torch.tensor(ref_mask, dtype=torch.float32).unsqueeze(0)
        prev_mask = torch.tensor(prev_mask, dtype=torch.float32).unsqueeze(0)

        return ref_img, ref_mask, cur_img, cur_mask, prev_mask, crop_box
