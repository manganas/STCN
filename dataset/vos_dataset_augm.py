import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed

# from vos_augmentation.vos_augmentation import VOS_Augmentation


class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """

    def __init__(self, im_root, gt_root, max_jump, is_bl, subset=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl

        self.videos = []
        self.frames = {}

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < 3:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print(
            "%d out of %d videos accepted in %s."
            % (len(self.videos), len(vid_list), im_root)
        )

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.01, 0.01, 0.01, 0),
            ]
        )

        self.pair_im_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=15,
                    shear=10,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=im_mean,
                ),
            ]
        )

        self.pair_gt_dual_transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=15,
                    shear=10,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                ),
            ]
        )

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose(
            [
                transforms.ColorJitter(0.1, 0.03, 0.03, 0),
                transforms.RandomGrayscale(0.05),
            ]
        )

        if self.is_bl:
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (384, 384),
                        scale=(0.25, 1.00),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                ]
            )

            self.all_gt_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (384, 384),
                        scale=(0.25, 1.00),
                        interpolation=InterpolationMode.NEAREST,
                    ),
                ]
            )
        else:
            self.all_im_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (384, 384),
                        scale=(0.36, 1.00),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                ]
            )

            self.all_gt_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (384, 384),
                        scale=(0.36, 1.00),
                        interpolation=InterpolationMode.NEAREST,
                    ),
                ]
            )

        # Final transform without randomness
        self.final_im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                im_normalization,
            ]
        )

    def __get_data__(self, idx):
        video = self.videos[idx]
        info = {}
        info["name"] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info["frames"] = []  # Appended with actual frames

            # Don't want to bias towards beginning/end
            this_max_jump = min(len(frames), self.max_jump)
            start_idx = np.random.randint(len(frames) - this_max_jump + 1)
            f1_idx = start_idx + np.random.randint(this_max_jump + 1) + 1
            f1_idx = min(f1_idx, len(frames) - this_max_jump, len(frames) - 1)

            f2_idx = f1_idx + np.random.randint(this_max_jump + 1) + 1
            f2_idx = min(f2_idx, len(frames) - this_max_jump // 2, len(frames) - 1)

            frames_idx = [start_idx, f1_idx, f2_idx]

            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None

            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + ".jpg"
                png_name = frames[f_idx][:-4] + ".png"
                info["frames"].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert("RGB")
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert("P")
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels != 0]

            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (masks[0] == l).sum()
                    if pixel_sum > 10 * 10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30 * 30:
                            good_lables.append(l)
                        elif (
                            max((masks[1] == l).sum(), (masks[2] == l).sum()) < 20 * 20
                        ):
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)

            if len(labels) == 0:
                target_object = -1  # all black if no objects
                has_second_object = False
                trials += 1
            else:
                target_object = np.random.choice(labels)
                has_second_object = len(labels) > 1
                if has_second_object:
                    labels = labels[labels != target_object]
                    second_object = np.random.choice(labels)
                break

        masks = np.stack(masks, 0)
        tar_masks = (masks == target_object).astype(np.float32)[:, np.newaxis, :, :]
        if has_second_object:
            sec_masks = (masks == second_object).astype(np.float32)[:, np.newaxis, :, :]
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((3, 384, 384), dtype=int)
        cls_gt[tar_masks[:, 0] > 0.5] = 1
        cls_gt[sec_masks[:, 0] > 0.5] = 2

        data = {
            "rgb": images,
            "gt": tar_masks,
            "cls_gt": cls_gt,
            "sec_gt": sec_masks,
            "selector": selector,
            "info": info,
        }

        return data

    def __getitem__(self, idx):
        """
        data = {
            "rgb": images,
            "gt": tar_masks,
            "cls_gt": cls_gt,
            "sec_gt": sec_masks,
            "selector": selector,
            "info": info,
        }
        """

        original_data = self.__get_data__(idx)

        # Get another video
        j = np.random.randint(low=0, high=len(self.videos))
        while j == idx:
            j = np.random.randint(low=0, high=len(self.videos))

        new_data = self.__get_data__(j)

        og_frames = original_data["rgb"]
        og_masks = original_data["gt"]
        og_tar_masks = original_data["cls_gt"]
        og_sec_masks = original_data["sec_gt"]

        new_frames = new_data["rgb"]
        new_masks = new_data["gt"]

        augm_frames = []
        augm_masks = []

        ## Problem if not davis@ not all frames have masks!
        # for i in range(og_frames.shape[0]):
        #     augm_frame = og_frames[i].permute(1, 2, 0)
        #     augm_mask = og_masks[i].permute(1, 2, 0)

        #     new_frame = new_frames[i].permute(1, 2, 0)
        #     new_mask = new_masks[i].permute(1, 2, 0)

        #     augm_frame[new_mask > 0] = new_frame[new_mask > 0]
        #     augm_mask[new_mask > 0] = 0.0

        #     augm_frames.append(augm_frame.permute(2, 0, 1))
        #     augm_masks.append(augm_mask.permute(2, 0, 1))

        #     # og_tar_masks[i][new_masks[i]>0] = 0
        #     # og_sec_masks[i][new_masks[i]>0] = 0

        # augmented_data = {
        #     "rgb": augm_frames,
        #     "gt": augm_masks,
        #     # "cls_gt": cls_gt,
        #     # "sec_gt": sec_masks,
        #     # "selector": selector,
        #     # "info": info,
        # }

        return original_data

    # def __getitem_OLD__(self, idx):

    #     video = self.videos[idx]
    #     info = {}
    #     info["name"] = video

    #     vid_im_path = path.join(self.im_root, video)
    #     vid_gt_path = path.join(self.gt_root, video)
    #     frames = self.frames[video]

    #     # Get another video
    #     j = np.random.randint(low=0, high=len(self.videos))
    #     while j == idx:
    #         j = np.random.randint(low=0, high=len(self.videos))

    #     new_video = self.videos[j]
    #     new_vid_im_path = path.join(self.im_root, new_video)
    #     new_vid_gt_path = path.join(self.gt_root, new_video)
    #     new_frames = self.frames[new_video]

    #     trials = 0
    #     while trials < 5:
    #         info["frames"] = []  # Appended with actual frames

    #         # Don't want to bias towards beginning/end
    #         this_max_jump = min(len(frames), self.max_jump)
    #         start_idx = np.random.randint(len(frames) - this_max_jump + 1)
    #         f1_idx = start_idx + np.random.randint(this_max_jump + 1) + 1
    #         f1_idx = min(f1_idx, len(frames) - this_max_jump, len(frames) - 1)

    #         f2_idx = f1_idx + np.random.randint(this_max_jump + 1) + 1
    #         f2_idx = min(f2_idx, len(frames) - this_max_jump // 2, len(frames) - 1)

    #         frames_idx = [start_idx, f1_idx, f2_idx]

    #         ### Augmentation ###
    #         ####################
    #         new_this_max_jump = min(len(new_frames), self.max_jump)
    #         new_start_idx = np.random.randint(len(new_frames) - new_this_max_jump + 1)
    #         new_f1_idx = new_start_idx + np.random.randint(new_this_max_jump + 1) + 1
    #         new_f1_idx = min(
    #             new_f1_idx, len(new_frames) - new_this_max_jump, len(new_frames) - 1
    #         )

    #         new_f2_idx = new_f1_idx + np.random.randint(new_this_max_jump + 1) + 1
    #         new_f2_idx = min(
    #             new_f2_idx,
    #             len(new_frames) - new_this_max_jump // 2,
    #             len(new_frames) - 1,
    #         )

    #         new_frames_idx = [new_start_idx, new_f1_idx, new_f2_idx]
    #         ####################
    #         ####################

    #         if np.random.rand() < 0.5:
    #             # Reverse time
    #             frames_idx = frames_idx[::-1]

    #         ### Augmentation ###
    #         ####################
    #         if np.random.rand() < 0.5:
    #             # Reverse time for augmentation frame
    #             new_frames_idx = new_frames_idx[::-1]
    #         ####################
    #         ####################

    #         sequence_seed = np.random.randint(2147483647)
    #         images = []
    #         masks = []
    #         target_object = None

    #         ### Augmentation ###
    #         ####################
    #         new_images = []
    #         new_masks = []
    #         ####################
    #         ####################

    #         for f_idx in frames_idx:
    #             jpg_name = frames[f_idx][:-4] + ".jpg"
    #             png_name = frames[f_idx][:-4] + ".png"
    #             info["frames"].append(jpg_name)

    #             reseed(sequence_seed)
    #             this_im = Image.open(path.join(vid_im_path, jpg_name)).convert("RGB")
    #             this_im = self.all_im_dual_transform(this_im)
    #             this_im = self.all_im_lone_transform(this_im)
    #             reseed(sequence_seed)
    #             this_gt = Image.open(path.join(vid_gt_path, png_name)).convert("P")
    #             this_gt = self.all_gt_dual_transform(this_gt)

    #             pairwise_seed = np.random.randint(2147483647)
    #             reseed(pairwise_seed)
    #             this_im = self.pair_im_dual_transform(this_im)
    #             this_im = self.pair_im_lone_transform(this_im)
    #             reseed(pairwise_seed)
    #             this_gt = self.pair_gt_dual_transform(this_gt)

    #             this_im = self.final_im_transform(this_im)
    #             this_gt = np.array(this_gt)

    #             images.append(this_im)
    #             masks.append(this_gt)

    #         images = torch.stack(images, 0)

    #         labels = np.unique(masks[0])
    #         # Remove background
    #         labels = labels[labels != 0]

    #         if self.is_bl:
    #             # Find large enough labels
    #             good_lables = []
    #             for l in labels:
    #                 pixel_sum = (masks[0] == l).sum()
    #                 if pixel_sum > 10 * 10:
    #                     # OK if the object is always this small
    #                     # Not OK if it is actually much bigger
    #                     if pixel_sum > 30 * 30:
    #                         good_lables.append(l)
    #                     elif (
    #                         max((masks[1] == l).sum(), (masks[2] == l).sum()) < 20 * 20
    #                     ):
    #                         good_lables.append(l)
    #             labels = np.array(good_lables, dtype=np.uint8)

    #         if len(labels) == 0:
    #             target_object = -1  # all black if no objects
    #             has_second_object = False
    #             trials += 1
    #         else:
    #             target_object = np.random.choice(labels)
    #             has_second_object = len(labels) > 1
    #             if has_second_object:
    #                 labels = labels[labels != target_object]
    #                 second_object = np.random.choice(labels)
    #             break

    #     masks = np.stack(masks, 0)
    #     tar_masks = (masks == target_object).astype(np.float32)[:, np.newaxis, :, :]
    #     if has_second_object:
    #         sec_masks = (masks == second_object).astype(np.float32)[:, np.newaxis, :, :]
    #         selector = torch.FloatTensor([1, 1])
    #     else:
    #         sec_masks = np.zeros_like(tar_masks)
    #         selector = torch.FloatTensor([1, 0])

    #     cls_gt = np.zeros((3, 384, 384), dtype=int)
    #     cls_gt[tar_masks[:, 0] > 0.5] = 1
    #     cls_gt[sec_masks[:, 0] > 0.5] = 2

    #     ## Original data
    #     data = {
    #         "rgb": images,
    #         "gt": tar_masks,
    #         "cls_gt": cls_gt,
    #         "sec_gt": sec_masks,
    #         "selector": selector,
    #         "info": info,
    #     }

    #     ## augmented data dict
    #     augmented_data = self.vos_augment(original_data=data, new_video_idx=j)

    #     return data

    def __len__(self):
        return len(self.videos)
