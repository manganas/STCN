"""
Modified from https://github.com/seoungwugoh/STM/blob/master/dataset.py
"""

import os
from os import path
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data.dataset import Dataset
from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot

from pathlib import Path


class YouTubeTestDataset(Dataset):
    def __init__(
        self,
        root,
        imset="util/yv_subset.txt",
        resolution=480,
        single_object=False,
        target_name=None,
        steps: int = 1,
    ):
        self.root = Path(root)

        self.mask_dir = self.root.joinpath("train_480p", "Annotations")
        self.mask480_dir = self.root.joinpath("train_480p", "Annotations")
        self.image_dir = self.root.joinpath("train_480p", "JPEGImages")
        self.resolution = resolution

        all_video_names = sorted(list(self.mask480_dir.iterdir()))
        all_video_names = [vid.stem for vid in all_video_names]

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(imset, "r") as lines:
            for line in lines:
                _video = line.rstrip("\n")
                if target_name is not None and target_name != _video:
                    continue
                if _video not in all_video_names:
                    self.videos.append(_video)

                    frames = sorted(list(self.mask_dir.joinpath(_video).iterdir()))
                    self.num_frames[_video] = len(frames)
                    _mask = np.array(
                        Image.open(self.mask_dir.joinpath(_video, "00000.png")).convert(
                            "P"
                        )
                    )
                    self.num_objects[_video] = np.max(_mask)
                    self.shape[_video] = np.shape(_mask)
                    _mask480 = np.array(
                        Image.open(
                            self.mask480_dir.joinpath(_video, "00000.png")
                        ).convert("P")
                    )
                    self.size_480p[_video] = np.shape(_mask480)

        self.single_object = single_object

        self.step = steps

        if resolution == 480:
            self.im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    im_normalization,
                ]
            )
        else:
            self.im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    im_normalization,
                    transforms.Resize(
                        resolution, interpolation=InterpolationMode.BICUBIC
                    ),
                ]
            )
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(
                        resolution, interpolation=InterpolationMode.NEAREST
                    ),
                ]
            )

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info["name"] = video
        info["frames"] = []
        info["num_frames"] = self.num_frames[video]
        info["size_480p"] = self.size_480p[video]

        images = []
        masks = []

        frames = sorted(list(self.mask480_dir.joinpath(video).iterdir()))
        frame_names = [frame.stem for frame in frames]

        print()
        print(len(frame_names))
        print()
        for f in range(0, self.num_frames[video], self.step):
            img_file = path.join(
                self.image_dir, video, "{:05d}.jpg".format(frame_names[f])
            )
            images.append(self.im_transform(Image.open(img_file).convert("RGB")))
            info["frames"].append("{:05d}.jpg".format(f))

            mask_file = path.join(
                self.mask_dir, video, "{:05d}.png".format(frame_names[f])
            )
            if path.exists(mask_file):
                masks.append(
                    np.array(Image.open(mask_file).convert("P"), dtype=np.uint8)
                )
            else:
                # Test-set maybe?
                print("Online validation mask not found")
                masks.append(np.zeros_like(masks[0]))

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)

        if self.single_object:
            labels = [1]
            masks = (masks > 0.5).astype(np.uint8)
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()
        else:
            labels = np.unique(masks[0])
            labels = labels[labels != 0]
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        if self.resolution != 480:
            masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info["labels"] = labels

        data = {
            "rgb": images,
            "gt": masks,
            "info": info,
        }

        return data
