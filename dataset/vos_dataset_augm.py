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

        self.train = train

        ## Augmentation haparams
        # Probs
        augmentation_params = para["augmentations"]

        self.p_augm = augmentation_params["augmentation_p"]
        print(self.p_augm)

        p_horizontal_flip = augmentation_params["horizontal_flip_p"]
        p_scale = augmentation_params["scale_p"]
        p_translation = augmentation_params["translation_p"]

        # Params
        self.scale_factor_lim = augmentation_params["scale_factor_lim"]
        self.translation_lim = augmentation_params["translation_lim"]

        self.vos_augmentations = VOSAugmentations(
            select_instances=False, always_foreground=True, include_new_instances=False
        )

        self.transformations_list = [
            VOSTransformations.random_horizontal_flip(p_horizontal_flip),
            VOSTransformations.random_scale(p_scale),
            VOSTransformations.random_translation(p_translation),
        ]

    def __get_data__(self, idx):
        video = self.videos[idx]
        info = {}
        info["name"] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        augment = False
        if self.train and np.random.rand() < self.p_augm:
            augment = True
            j = np.random.randint(low=0, high=len(self.videos))
            augm_video = self.videos[j]
            augm_vid_im_path = path.join(self.im_root, augm_video)
            augm_vid_gt_path = path.join(self.gt_root, augm_video)
            augm_frames = self.frames[augm_video]

        # if augment:
        #     print("Augment")
        # else:
        #     print("Not augment")

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

            ## Augmentation jumps
            if augment:
                this_max_jump = min(len(augm_frames), self.max_jump)
                start_idx = np.random.randint(len(augm_frames) - this_max_jump + 1)
                f1_idx = start_idx + np.random.randint(this_max_jump + 1) + 1
                f1_idx = min(
                    f1_idx, len(augm_frames) - this_max_jump, len(augm_frames) - 1
                )

                f2_idx = f1_idx + np.random.randint(this_max_jump + 1) + 1
                f2_idx = min(
                    f2_idx, len(augm_frames) - this_max_jump // 2, len(augm_frames) - 1
                )

                augm_frames_idx = [start_idx, f1_idx, f2_idx]

                if np.random.rand() < 0.5:
                    # Reverse time
                    augm_frames_idx = augm_frames_idx[::-1]

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

                    # Transforms to be applied to the new mask
                    horizontal_p = np.random.rand()

                    scale_p = np.random.rand()
                    scale_factor = (
                        np.random.randint(
                            low=-self.scale_factor_lim, high=self.scale_factor_lim
                        )
                        * 0.01
                    )

                    translation_p = np.random.rand()
                    translation_w_lim = np.random.randint(
                        low=-int(384 * self.translation_lim),
                        high=int(384 * self.translation_lim),
                    )

                    translation_h_lim = np.random.randint(
                        low=-int(384 * self.translation_lim),
                        high=int(384 * self.translation_lim),
                    )

                    transformation_options = {
                        "resize_w": 384,
                        "resize_h": 384,
                        "translation_p": translation_p,
                        "translation": (translation_w_lim, translation_h_lim),
                        "horizontal_p": horizontal_p,
                        "scale_p": scale_p,
                        "scale_factor": scale_factor,
                    }

                    this_im, this_gt = (
                        self.vos_augmentations.get_augmented_data_per_frame(
                            this_im,
                            this_gt,
                            that_im,
                            that_gt,
                            self.transformations_list,
                            **transformation_options
                        )
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

        data = self.__get_data__(idx)

        # # Get another video, if p>0.5
        # seed = np.random.randint(2147483647)
        # reseed(seed)
        # p = np.random.rand()
        # if p >= 0.5:
        #     j = np.random.randint(low=0, high=len(self.videos))

        #     new_data = self.__get_data__(j)

        #     data = VOSAugmentations.get_augmented_data(data, new_data, [])

        ## Idea: Propta fortono kai augment eikones, san PIL.Image
        # Meta kano ola ta alla gia augmented mask aki frames: dual transf etc..
        # poy edo ta kanei otan travaei ta dedomena.
        # Ligi prosoxi an tha allaksei to selector, alla vlepoyme

        return data

    def __len__(self):
        return len(self.videos)
