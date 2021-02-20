import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def get_palette(num_objects):
    def uint8_to_bin(n):
        """returns the binary of integer n"""
        return [str((n >> i) & 1) for i in reversed(range(8))]

    cmap = np.zeros((num_objects, 3), dtype=np.uint8)
    for i in range(num_objects):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint8_to_bin(id)
            r ^= np.uint8(str_id[-1]) << (7 - j)
            g ^= np.uint8(str_id[-2]) << (7 - j)
            b ^= np.uint8(str_id[-3]) << (7 - j)
            id >>= 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in-dir', type=str, required=True)
    parser.add_argument('-o', '--out-dir', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['soft', 'hard'], default='soft')
    args = parser.parse_args()

    palette = get_palette(256)

    in_dir = Path(args.in_dir)
    assert in_dir.is_dir(), f'{in_dir} is not a directory'
    print(f'Merging masks in {in_dir}')

    for video_dir in tqdm(sorted(in_dir.glob('*'))):
        assert video_dir.is_dir(), f'{video_dir} is not a directory'
        object_indices = [x.name for x in video_dir.glob('*')]

        mask_names = [x.name for x in video_dir.glob('*/*')]
        mask_names = sorted(list(set(mask_names)))

        for mask_name in mask_names:
            scores = None
            for score_index, object_index in enumerate(object_indices):
                mask_path = video_dir / object_index / mask_name
                if not mask_path.is_file():
                    continue

                cur_score = Image.open(str(mask_path)).convert('L')
                cur_score = np.array(cur_score).astype(np.float32) / 255

                if scores is None:
                    scores = np.zeros((1 + len(object_indices), *cur_score.shape), dtype=np.float32)
                    scores[0] = 1.0

                # object score
                scores[score_index + 1] = cur_score

                if args.mode == 'soft':
                    # soft aggregation
                    scores[0] *= 1 - cur_score
                elif args.mode == 'hard':
                    scores[0] = np.minimum(scores[0], 1 - cur_score)
                else:
                    raise ValueError(f'Invalid merge mode {args.mode}')

            merged_mask = np.argmax(scores, axis=0).astype(np.int32)
            merged_mask = np.array([0, *object_indices], dtype=np.uint8)[merged_mask]

            save_dir = Path(args.out_dir) / video_dir.name
            save_dir.mkdir(parents=True, exist_ok=True)

            save_img = Image.fromarray(merged_mask)
            save_img.putpalette(palette)
            save_img.save(save_dir / mask_name)


if __name__ == '__main__':
    main()
