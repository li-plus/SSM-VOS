import random

import cv2
import numpy as np


def rotate_bound(image, angle, interpolation):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=interpolation)


def compute_robust_moments(binary_image, isotropic=False):
    index = np.nonzero(binary_image)
    points = np.asarray(index).astype(np.float32)
    if points.shape[1] == 0:
        return np.array([-1.0, -1.0], dtype=np.float32), np.array([-1.0, -1.0], dtype=np.float32)
    points = np.transpose(points)
    points[:, [0, 1]] = points[:, [1, 0]]
    center = np.median(points, axis=0)
    if isotropic:
        diff = np.linalg.norm(points - center, axis=1)
        mad = np.median(diff)
        mad = np.array([mad, mad])
    else:
        diff = np.absolute(points - center)
        mad = np.median(diff, axis=0)
    std_dev = 1.4826 * mad
    std_dev = np.maximum(std_dev, [5.0, 5.0])
    return center, std_dev


def get_gb_image(label, center_perturb=0.2, std_perturb=0.4, blank_prob=0):
    label = np.array(label)
    if not np.any(label) or random.random() < blank_prob:
        return np.zeros_like(label)
    center, std = compute_robust_moments(label)
    center_p_ratio = np.random.uniform(-center_perturb, center_perturb, 2)
    center_p = center_p_ratio * std + center
    std_p_ratio = np.random.uniform(
        1.0 / (1 + std_perturb), 1.0 + std_perturb, 2)
    std_p = std_p_ratio * std
    h, w = label.shape
    x = np.arange(0, w)
    y = np.arange(0, h)
    nx, ny = np.meshgrid(x, y)
    coords = np.concatenate((nx[..., np.newaxis], ny[..., np.newaxis]), axis=2)
    normalizer = 0.5 / (std_p * std_p)
    D = np.sum((coords - center_p) ** 2 * normalizer, axis=2)
    D = np.exp(-D)
    D = np.clip(D, 0, 1)
    return D


def crop_array(array, box_min, box_max, dim, pad_value=0):
    """Crop the array at arbitrary dimension, with padding if min < 0 or max > range."""
    array = np.array(array)
    box_max = int(box_max)
    box_min = int(box_min)
    assert box_min < box_max

    new_shape = list(array.shape)
    new_shape[dim] = box_max - box_min
    array_crop = np.full(new_shape, pad_value, dtype=array.dtype)
    pad_left = max(0, -box_min)

    roi_left = np.clip(box_min, 0, array.shape[dim])
    roi_right = np.clip(box_max, 0, array.shape[dim])
    roi_width = roi_right - roi_left

    crop_index = [slice(None) for _ in range(array.ndim)]
    crop_index[dim] = slice(pad_left, pad_left + roi_width)

    roi_index = [slice(None) for _ in range(array.ndim)]
    roi_index[dim] = slice(roi_left, roi_right)

    array_crop[tuple(crop_index)] = array[tuple(roi_index)]

    return array_crop


def crop_image(image, crop_box):
    x1, y1, x2, y2 = crop_box
    out = image
    out = crop_array(out, y1, y2, dim=0)
    out = crop_array(out, x1, x2, dim=1)
    return out


def crop_image_back(image, old_crop_box, org_h, org_w):
    x1 = -old_crop_box[0]
    y1 = -old_crop_box[1]
    x2 = x1 + org_w
    y2 = y1 + org_h
    return crop_image(image, [x1, y1, x2, y2])
