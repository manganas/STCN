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

from dataset.data_augmentations import (
    VOSAugmentations,
    VOSTransformations,
    get_n_successive_augm,
    calculate_nested_probabilities,
)
from dataset.data_augmentations_coco import CocoAugmentor

from dataset.augmentations.base_augmentor import Augmentor


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

        p_augm_before = sorted(
            augmentation_params["augmentation_p"], reverse=True
        )  # From highest to lowest probs

        self.p_augm = calculate_nested_probabilities(p_augm_before)

        p_horizontal_flip = augmentation_params["horizontal_flip_p"]

        # Params
        self.scale_factor_lower_lim = augmentation_params["scale_factor_lower_lim"]
        self.scale_factor_upper_lim = augmentation_params["scale_factor_upper_lim"]
        self.translation_lim = augmentation_params["translation_lim"]

        self.transformations_list = [
            VOSTransformations.random_horizontal_flip(p_horizontal_flip),
            VOSTransformations.random_scale,
            VOSTransformations.random_translation,
        ]

        # Augmentors and datasets
        datasets = augmentation_params["augmentation_datasets"]
        self.augmentor = Augmentor(
            datasets=datasets,
            probabilities=None,
        )

        self.vos_augmentations = VOSAugmentations(
            select_instances=augmentation_params["select_instances"],
            foreground_p=augmentation_params["foreground_p"],
            include_new_instances=augmentation_params["include_new_instances"],
        )

        # better to instantiate more augmentor objects here
        # than to dynamically instantiate in the call of __getitem__

        # self.vos_augmentations = []
        # for _ in range(len(self.p_augm)):
        #     self.vos_augmentations.append(
        #         VOSAugmentations(
        #             select_instances=augmentation_params["select_instances"],
        #             foreground_p=augmentation_params["foreground_p"],
        #             include_new_instances=augmentation_params["include_new_instances"],
        #         )
        #     )

        # self.use_coco = False
        # if (
        #     "use_coco" in list(augmentation_params.keys())
        #     and augmentation_params["use_coco"]
        # ):
        #     self.use_coco = True
        #     davis_root_path = para["davis_root"]
        #     davis_path = os.path.join(davis_root_path, "2017", "trainval")
        #     # self.coco_augmentor = CocoAugmentor(para["coco_root"], davis_path)
        #     self.coco_augmentor = []
        #     for _ in range(len(self.p_augm)):
        #         self.coco_augmentor.append(CocoAugmentor(para["coco_root"], davis_path))

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

        # Number of successive augmentations for this __getitem__ call
        if self.train:
            n_augmentations = get_n_successive_augm(self.p_augm)

            # Select from which datasets the augmentation data will come from
            # eg. selected_datasets = ['coco', 'fss', 'davis/yt']
            augmentation_datasets = self.augmentor.select_datasets(
                len(n_augmentations), replace=True
            )

            self.augmentor.set_augmentors()

            # #################
            # #################
            # #################
            # augm_vid_im_paths = [None] * len(n_augmentations)
            # augm_vid_gt_paths = [None] * len(n_augmentations)
            # augm_frames_list = [None] * len(n_augmentations)

            # if self.use_coco:
            #     coco_frames_lists = [None] * len(n_augmentations)
            #     coco_masks_lists = [None] * len(n_augmentations)
            # #################
            # #################
            # #################

        else:
            n_augmentations = []

        for augmentor_idx in range(len(n_augmentations)):
            # j = np.random.randint(low=0, high=len(self.videos))

            # augm_vid_im_path, augm_vid_gt_path, augm_frames = self.get_vid_frames_paths(
            #     j
            # )

            # augm_vid_im_paths[augmentor_idx] = augm_vid_im_path
            # augm_vid_gt_paths[augmentor_idx] = augm_vid_gt_path
            # augm_frames_list[augmentor_idx] = augm_frames

            # could be more, but I only select 1 additional class from all the newer frames. Here I add 3 inst of 1
            self.vos_augmentations.max_n_classes_per_frame += (augmentor_idx + 1) * 3

            # if self.use_coco:
            #     coco_frames_list, coco_masks_list = self.coco_augmentor[
            #         augmentor_idx
            #     ].get_augmentation_data(self.videos[j])

            #     coco_frames_lists[augmentor_idx] = coco_frames_list
            #     coco_masks_lists[augmentor_idx] = coco_masks_list

        self.vos_augmentations.reset_chosen_instances()

        trials = 0
        while trials < 5:
            info["frames"] = []  # Appended with actual frames
            # Don't want to bias towards beginning/end
            frames_idx = self.get_frames_indices(frames)

            # This random reversal can be included inside the get_frames_indices to avoid repeating
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            augm_frames_indices = []
            # for augm_idx in range(len(n_augmentations)):
            # augm_frames_idx = self.get_frames_indices(augm_frames_list[augm_idx])

            # if np.random.rand() < 0.5:
            #     # Reverse time
            #     augm_frames_idx = augm_frames_idx[::-1]

            # augm_frames_indices.append(augm_frames_idx)

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []

            # for intermediate augmentations
            images_ = []
            masks_ = []
            target_object = None

            # for f_idx in frames_idx:
            for i in range(len(frames_idx)):
                f_idx = frames_idx[i]

                jpg_name = frames[f_idx][:-4] + ".jpg"
                png_name = frames[f_idx][:-4] + ".png"
                info["frames"].append(jpg_name)

                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert("RGB")
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert("P")

                images_.append(this_im)
                masks_.append(this_gt)

                ###

            for augment_idx in range(len(n_augmentations)):
                # augmentor = self.augmentor.augmentors[augment_idx]
                self.vos_augmentations.set_seed(100 * idx + augmentor_idx)
                self.vos_augmentations.reset_chosen_instances()

                new_images, new_masks = self.augmentor.get_augmentation_data(
                    augment_idx
                )
                for i in range(len(frames_idx)):
                    # augm_f_idx = augm_frames_indices[augment_idx][i]

                    # augm_jpg_name = (
                    #     augm_frames_list[augment_idx][augm_f_idx][:-4] + ".jpg"
                    # )
                    # augm_png_name = (
                    #     augm_frames_list[augment_idx][augm_f_idx][:-4] + ".png"
                    # )

                    that_im = new_images[i]
                    that_gt = new_masks[i]

                    # that_im = Image.open(new_images_paths[i]).convert("RGB")
                    # that_gt = Image.open(new_masks_paths[i]).convert("P")

                    # if self.use_coco:
                    #     that_im, that_gt = (
                    #         coco_frames_lists[augment_idx][augm_f_idx],
                    #         coco_masks_lists[augment_idx][augm_f_idx],
                    #     )

                    this_im, this_gt = self.augment_image(
                        images_[i],
                        masks_[i],
                        that_im,
                        that_gt,
                    )

                    images_[i] = this_im
                    masks_[i] = this_gt

                    # remember to reset augmentor. Use an overriden method for that
                    # for davis only. Others empty
                    # augmentor.reset()
                ###

            for i in range(len(frames_idx)):
                this_im = images_[i]
                this_gt = masks_[i]

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
