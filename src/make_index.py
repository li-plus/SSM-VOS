import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

DAVIS16_TRAIN_VAL_ROOT = '../datasets/DAVIS2016/'
DAVIS17_TRAIN_VAL_ROOT = '../datasets/DAVIS2017/'
DAVIS17_TEST_ROOT = '../datasets/DAVIS2017_test/'
YOUTUBE_ROOT = '../datasets/YouTubeVOS/'


def index_youtube(split):
    assert split in ['train', 'test']
    if split == 'train':
        dataset_root = Path(YOUTUBE_ROOT) / 'train'
    else:
        dataset_root = Path(YOUTUBE_ROOT) / 'valid'
    assert dataset_root.is_dir()

    video_root = dataset_root / 'JPEGImages'
    assert video_root.is_dir()
    label_root = dataset_root / 'Annotations'
    assert label_root.is_dir()
    meta_path = dataset_root / 'meta.json'
    assert meta_path.is_file()

    with open(meta_path) as f:
        meta = json.load(f)

    split_info = []

    for video_name, video in tqdm(meta['videos'].items()):
        for object_index, obj in video['objects'].items():
            mask_path = None

            for frame_index, frame in enumerate(obj['frames']):
                object_index = int(object_index)
                image_path = video_root / video_name / f'{frame}.jpg'

                if split != 'test' or frame_index == 0:
                    mask_path = label_root / video_name / f'{frame}.png'

                num_frames = len(obj['frames'])

                video_info = dict(image_path=str(image_path.resolve()),
                                  mask_path=str(mask_path.resolve()),
                                  video_name=video_name,
                                  num_frames=num_frames,
                                  frame_index=frame_index,
                                  object_index=object_index)
                split_info.append(video_info)

    return split_info


def index_davis(video_root, label_root, split, video_names):
    split_info = []

    for video_name in tqdm(video_names):
        video_dir = video_root / video_name
        assert video_dir.is_dir()
        label_dir = label_root / video_name
        assert label_dir.is_dir()

        img_paths = sorted(video_dir.glob('*.jpg'))
        mask_paths = sorted(label_dir.glob('*.png'))

        if split == 'test':
            assert len(mask_paths) == 1
            mask_paths *= len(img_paths)
        else:
            assert len(img_paths) == len(mask_paths)
            assert [f.stem for f in img_paths] == [f.stem for f in mask_paths]

        mask_first = np.array(Image.open(mask_paths[0]), dtype=np.uint8)
        object_indices = [int(x) for x in np.unique(mask_first) if x != 0]

        num_frames = len(img_paths)

        for object_index in object_indices:
            video_info = [dict(image_path=str(image_path.resolve()),
                               mask_path=str(mask_path.resolve()),
                               video_name=video_name,
                               num_frames=num_frames,
                               frame_index=frame_index,
                               object_index=object_index)
                          for frame_index, (image_path, mask_path) in enumerate(zip(img_paths, mask_paths))]

            split_info += video_info

    return split_info


def index_davis17(split):
    assert split in ['train', 'val', 'test']
    if split == 'test':
        dataset_root = Path(DAVIS17_TEST_ROOT)
    else:
        dataset_root = Path(DAVIS17_TRAIN_VAL_ROOT)
    assert dataset_root.is_dir()
    meta_name_map = {
        'train': 'train.txt',
        'val': 'val.txt',
        'test': 'test-dev.txt'
    }
    meta_path = dataset_root / 'ImageSets/2017' / meta_name_map[split]
    with open(meta_path) as f:
        video_names = f.read().split()

    video_root = dataset_root / 'JPEGImages/480p'
    label_root = dataset_root / 'Annotations/480p'
    return index_davis(video_root, label_root, split, video_names)


def index_davis16(split):
    assert split in ['train', 'val']
    dataset_root = Path(DAVIS16_TRAIN_VAL_ROOT)
    assert dataset_root.is_dir()
    video_root = dataset_root / 'JPEGImages' / '480p'
    assert video_root.is_dir()
    label_root = dataset_root / 'Annotations' / '480p'
    assert label_root.is_dir()

    meta_path = dataset_root / 'ImageSets' / '480p' / f'{split}.txt'
    assert meta_path.is_file()
    with open(meta_path) as f:
        lines = f.readlines()

    video_names = set()
    for line in lines:
        img_path, _ = line.split()
        video_names.add(Path(img_path).parent.name)

    return index_davis(video_root, label_root, split, video_names)


def index_dataset(dataset, split, out_dir):
    assert dataset in ['davis2016', 'davis2017', 'youtube']
    print(f'Indexing {dataset}.{split}')
    if dataset == 'davis2016':
        split_info = index_davis16(split)
    elif dataset == 'davis2017':
        split_info = index_davis17(split)
    elif dataset == 'youtube':
        split_info = index_youtube(split)
    else:
        assert False

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f'{dataset}_{split}.json', 'w') as f:
        json.dump(split_info, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str, default='../splits')
    args = parser.parse_args()

    index_dataset('davis2016', 'train', args.out_dir)
    index_dataset('davis2016', 'val', args.out_dir)
    index_dataset('davis2017', 'train', args.out_dir)
    index_dataset('davis2017', 'val', args.out_dir)
    index_dataset('davis2017', 'test', args.out_dir)
    index_dataset('youtube', 'train', args.out_dir)
    index_dataset('youtube', 'test', args.out_dir)


if __name__ == '__main__':
    main()
