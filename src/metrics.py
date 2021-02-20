import numpy as np


def iou(mask_x, mask_y):
    mask_x = np.asarray(mask_x, dtype=np.bool)
    mask_y = np.asarray(mask_y, dtype=np.bool)

    union = np.sum(mask_x | mask_y)
    inter = np.sum(mask_x & mask_y)

    return float(inter / union) if union != 0 else 1.0
