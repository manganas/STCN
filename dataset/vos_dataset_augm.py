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

from dataset.data_augmentations import VOSAugmentations, VOSTransformations


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

    def __init__(
        self,
        im_root,
        gt_root,
        max_jump,
        is_bl,
        subset=None,
        train: bool = True,
        para=None,
    ):
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

        # Image dimensions
        self.resize_h = 384
        self.resize_w = 384

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
                        (self.resize_h, self.resize_w),
                        scale=(0.25, 1.00),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                ]
            )

            self.all_gt_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (self.resize_h, self.resize_w),
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
                        (self.resize_h, self.resize_w),
                        scale=(0.36, 1.00),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                ]
            )

            self.all_gt_dual_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(
                        (self.resize_h, self.resize_w),
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

        self.train = train

        ## Augmentation haparams
        # Probs
        augmentation_params = para["augmentations"]

        self.p_augm = augmentation_params["augmentation_p"]
        self.third_frame_p = augmentation_params["third_frame_p"]
        print(f"Augmentation p: {self.p_augm}")

        p_horizontal_flip = augmentation_params["horizontal_flip_p"]

        # Params
        self.scale_factor_lower_lim = augmentation_params["scale_factor_lower_lim"]
        self.scale_factor_upper_lim = augmentation_params["scale_factor_upper_lim"]
        self.translation_lim = augmentation_params["translation_lim"]

        self.vos_augmentations = VOSAugmentations(
            select_instances=augmentation_params["select_instances"],
            foreground_p=augmentation_params["foreground_p"],
            include_new_instances=augmentation_params["include_new_instances"],
        )

        self.transformations_list = [
            VOSTransformations.random_horizontal_flip(p_horizontal_flip),
            VOSTransformations.random_scale,
            VOSTransformations.random_translation,
        ]

        print(
            f"Augmentations, scaling: upper={self.scale_factor_upper_lim}, lower={self.scale_factor_lower_lim}"
        )
        print(f"Augmentations, translation: limit ratio={self.translation_lim}")
        print(f"Augmentations, horizontal flip prob: {p_horizontal_flip}")

    def get_vid_frames_paths(self, idx: int) -> list:
        video = self.videos[idx]
        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]
        return vid_im_path, vid_gt_path, frames

    def get_frames_indices(self, frames: list) -> list[int]:
        this_max_jump = min(len(frames), self.max_jump)
        start_idx = np.random.randint(len(frames) - this_max_jump + 1)
        f1_idx = start_idx + np.random.randint(this_max_jump + 1) + 1
        f1_idx = min(f1_idx, len(frames) - this_max_jump, len(frames) - 1)

        f2_idx = f1_idx + np.random.randint(this_max_jump + 1) + 1
        f2_idx = min(f2_idx, len(frames) - this_max_jump // 2, len(frames) - 1)

        frames_idx = [start_idx, f1_idx, f2_idx]
        return frames_idx

    def augment_image(
        self, original_im, original_gt, new_im, new_gt
    ) -> tuple[Image.Image, Image.Image]:
        # Transforms to be applied to the new mask
        horizontal_p = np.random.rand()

        scale_factor = np.random.uniform(
            low=self.scale_factor_lower_lim,
            high=self.scale_factor_upper_lim,
        )

        translation_w_lim, translation_h_lim = 0, 0
        if self.translation_lim > 0:
            translation_w_lim = np.random.randint(
                low=-int(self.resize_w * self.translation_lim),
                high=int(self.resize_w * self.translation_lim),
            )

            translation_h_lim = np.random.randint(
                low=-int(self.resize_h * self.translation_lim),
                high=int(self.resize_h * self.translation_lim),
            )

        transformation_options = {
            "resize_w": self.resize_w,
            "resize_h": self.resize_h,
            "translation": (translation_w_lim, translation_h_lim),
            "horizontal_p": horizontal_p,
            "scale_factor": scale_factor,
        }

        this_im, this_gt = self.vos_augmentations.get_augmented_data_per_frame(
            original_im,
            original_gt,
            new_im,
            new_gt,
            self.transformations_list,
            **transformation_options,
        )

        return this_im, this_gt

    def __get_data__(self, idx):
        video = self.videos[idx]
        info = {}
        info["name"] = video

        vid_im_path, vid_gt_path, frames = self.get_vid_frames_paths(idx)

        augment = False
        augment_more = False
        if self.train and np.random.rand() < self.p_augm:
            augment = True
            j = np.random.randint(low=0, high=len(self.videos))
            augm_vid_im_path, augm_vid_gt_path, augm_frames = self.get_vid_frames_paths(
                j
            )

            if np.random.rand() < self.third_frame_p:
                augment_more = True
                self.vos_augmentations.max_n_classes_per_frame = 12  # if include
                k = np.random.randint(low=0, high=len(self.videos))
                augm_vid_im_path_3, augm_vid_gt_path_3, augm_frames_3 = (
                    self.get_vid_frames_paths(k)
                )

            self.vos_augmentations.reset_chosen_instances()
            self.vos_augmentations.set_seed(100 * idx)  # if include

        trials = 0
        while trials < 5:
            info["frames"] = []  # Appended with actual frames
            # Don't want to bias towards beginning/end
            frames_idx = self.get_frames_indices(frames)

            # This random reversal can be included inside the get_frames_indices to avoid repeating
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            ## Augmentation jumps
            ## Augmentation 1
            if augment:
                augm_frames_idx = self.get_frames_indices(augm_frames)

                if np.random.rand() < 0.5:
                    # Reverse time
                    augm_frames_idx = augm_frames_idx[::-1]

                ## Augmentation 2
                if augment_more:
                    augm_frames_idx_3 = self.get_frames_indices(augm_frames_3)
                    if np.random.rand() < 0.5:
                        # Reverse time
                        augm_frames_idx_3 = augm_frames_idx_3[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None

            # for f_idx in frames_idx:
            for i in range(len(frames_idx)):
                f_idx = frames_idx[i]

                jpg_name = frames[f_idx][:-4] + ".jpg"
                png_name = frames[f_idx][:-4] + ".png"
                info["frames"].append(jpg_name)

                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert("RGB")
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert("P")

                if augment:
                    augm_f_idx = augm_frames_idx[i]
                    augm_jpg_name = augm_frames[augm_f_idx][:-4] + ".jpg"
                    augm_png_name = augm_frames[augm_f_idx][:-4] + ".png"

                    that_im = Image.open(
                        path.join(augm_vid_im_path, augm_jpg_name)
                    ).convert("RGB")
                    that_gt = Image.open(
                        path.join(augm_vid_gt_path, augm_png_name)
                    ).convert("P")

                    this_im, this_gt = self.augment_image(
                        this_im, this_gt, that_im, that_gt
                    )

                    if augment_more:
                        augm_f_idx = augm_frames_idx_3[i]
                        augm_jpg_name = augm_frames_3[augm_f_idx][:-4] + ".jpg"
                        augm_png_name = augm_frames_3[augm_f_idx][:-4] + ".png"

                        that_im = Image.open(
                            path.join(augm_vid_im_path_3, augm_jpg_name)
                        ).convert("RGB")

                        that_gt = Image.open(
                            path.join(augm_vid_gt_path_3, augm_png_name)
                        ).convert("P")

                        this_im, this_gt = self.augment_image(
                            this_im, this_gt, that_im, that_gt
                        )

                reseed(sequence_seed)
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)

                reseed(sequence_seed)
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

        cls_gt = np.zeros((3, self.resize_h, self.resize_w), dtype=int)
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

        data = self.__get_data__(idx)

        return data

    def __len__(self):
        return len(self.videos)
