from PIL import Image
import numpy as np
from numpy.typing import ArrayLike

from typing import Callable
from numpy.typing import ArrayLike
from torch import Tensor

from PIL import Image
import numpy as np


class VOSTransformations:

    def random_scale(
        frame: ArrayLike | Tensor, mask: ArrayLike | Tensor, **kwargs
    ) -> ArrayLike | Tensor:
        """
        Resizes the image and crops the center part, sot that resulthas the same dims as input
        Scale is percentage increase or decrease: -0.99 -> 99% decrease in original dims
        """

        s = kwargs["scale_factor"]
        if s < 0.1:
            s = 0.1

        w, h = frame.size
        w_s = int((1 + s) * w)
        h_s = int((1 + s) * h)

        frame = frame.resize((w_s, h_s))
        mask = mask.resize((w_s, h_s))

        w_mid = w_s // 2
        h_mid = h_s // 2
        left, top, right, bottom = (
            w_mid - w // 2,
            h_mid - h // 2,
            w_mid + w // 2,
            h_mid + h // 2,
        )
        frame = frame.crop((left, top, right, bottom))
        mask = mask.crop((left, top, right, bottom))

        return frame, mask

    def random_horizontal_flip(p_horizontal_flip: float):
        def random_horizontal_flip_wrapper(
            frame: Image.Image, mask: Image.Image, **kwargs
        ) -> tuple[Image.Image]:

            p = kwargs["horizontal_p"]
            if p >= p_horizontal_flip:
                return frame, mask

            frame = frame.transpose(method=Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(method=Image.FLIP_LEFT_RIGHT)
            return frame, mask

        return random_horizontal_flip_wrapper

    def random_translation(
        frame: Image.Image, mask: Image.Image, **kwargs
    ) -> tuple[Image.Image]:
        """
        Random translation for the frame and the mask. Padded with zeros.
        It is used as a random positioning of the new image region.
        """

        translation = kwargs["translation"]

        return frame.rotate(0, translate=translation), mask.rotate(
            0, translate=translation
        )


class FrameCombiner:

    def __init__(
        self,
        select_instances: bool,
        foreground_p: float,
        include_new_instances: bool,
        max_n_classes_per_frame: int = 7,
    ) -> None:
        """
        foreground_p: probability that the augmentation mask is in the foreground, occluding the original mask.
        include_new_instances: If True, the selected instances in the augmentation mask will be included in the final mask
        select_instances: if True, in the case the augmentation maks has more than one instances, all or a subset of those will
                        be included in the augmentation to be applied.
        """

        if foreground_p > 1.0:
            self.foreground_p = 1.0
        elif foreground_p < 0.0:
            self.foreground_p = 0.0
        else:
            self.foreground_p = foreground_p

        self.select_instances = select_instances
        self.include_new_instances = include_new_instances

        self.max_n_classes_per_frame = max_n_classes_per_frame
        self.__chosen_instances = None

        self._in_foreground = np.random.rand() >= self.foreground_p

        print(f"Select instances: {self.select_instances}")
        print(f"Foreground probability: {self.foreground_p}")
        print(f"Include_new_instances: {self.include_new_instances}")

    def reset(self, max_n_classes_per_frame: int = 7) -> None:
        self.__chosen_instances = None
        self.max_n_classes_per_frame = max_n_classes_per_frame
        self._in_foreground = np.random.rand() >= self.foreground_p

    def reset_chosen_instances(self) -> None:
        self.__chosen_instances = None
        self._in_foreground = np.random.rand() >= self.foreground_p

    def apply_transformations(
        self,
        frame: Image.Image,
        mask: Image.Image,
        transformations_list: list = [],
        **kwargs,
    ) -> tuple[Image.Image]:
        for transform in transformations_list:
            frame, mask = transform(frame, mask, **kwargs)

        return frame, mask

    def select_mask_instances(self, mask_p: ArrayLike) -> ArrayLike:
        if self.__chosen_instances is None:
            discrete_inst = np.unique(mask_p)[1:]  # the 1st element is 0
            if len(discrete_inst) == 0:
                return mask_p
            try:
                n_inst = 1  # modification after meeting of 13/3/2024
            except ValueError:
                return mask_p
            chosen_instances = np.random.choice(
                discrete_inst, size=n_inst, replace=False
            )
            self.__chosen_instances = chosen_instances
        else:
            chosen_instances = self.__chosen_instances

        tmp_new_mask_p = np.zeros_like(mask_p)
        for i in sorted(chosen_instances):
            tmp_new_mask_p[mask_p == i] = i

            # Modif after meeting
            break

        return tmp_new_mask_p

    def get_new_mask_values(self, mask_new_p: ArrayLike) -> ArrayLike:
        """
        Assumption: The original mask is always in ordered form
        like [0,1,2,3,4]...
        This is the case when I first open the original mask as 'P' in PIL
        """

        new_mask_vals = np.unique(mask_new_p)[1:]  # ignore the background

        new_mask_copy = mask_new_p.copy()

        for i, el in enumerate(new_mask_vals):
            new_mask_copy[mask_new_p == el] = (
                new_mask_vals[i] + self.max_n_classes_per_frame
            )

        return new_mask_copy

    def overlay_frames_masks(
        self,
        frame_og: Image.Image,
        mask_og: Image.Image,
        frame_new: Image.Image,
        mask_new_p: Image.Image,
    ) -> tuple[Image.Image]:

        # Convert to numpy arrays
        frame_og = np.asarray(frame_og)
        mask_og_p = np.asarray(mask_og)
        frame_new = np.asarray(frame_new)
        mask_new_p = np.asarray(mask_new_p)

        # Select instances
        if self.select_instances:
            mask_new_p = self.select_mask_instances(mask_new_p)

        # Include new mask objects
        if self.include_new_instances:
            new_mask__ = self.get_new_mask_values(mask_new_p)
            # augmentation_mask_empty = 1 if len(np.where(new_mask__ > 0)[0]) < 50 else 0

        # Overlay mask of frame 2 on 1
        frame_augm = frame_og.copy()
        mask_augm = mask_og_p.copy()

        # back or foreground
        if self._in_foreground:
            frame_augm[mask_new_p > 0] = frame_new[mask_new_p > 0]
            if self.include_new_instances:
                mask_augm[new_mask__ > 0] = new_mask__[new_mask__ > 0]
            else:
                mask_augm[mask_new_p > 0] = 0
        else:
            if self.include_new_instances:
                tmp_new_mask = new_mask__.copy()
                tmp_new_mask[mask_og_p > 0] = 0

                mask_augm[tmp_new_mask > 0] = tmp_new_mask[tmp_new_mask > 0]

                frame_augm[tmp_new_mask > 0] = frame_new[tmp_new_mask > 0]
            else:
                tmp_new_mask = mask_new_p.copy()
                tmp_new_mask[mask_og_p > 0] = 0
                mask_augm[tmp_new_mask > 0] = 0
                frame_augm[tmp_new_mask > 0] = frame_new[tmp_new_mask > 0]

        # save frames and masks
        frame_augm = Image.fromarray(frame_augm)
        mask_augm = Image.fromarray(mask_augm)

        return frame_augm, mask_augm

    def get_augmented_data_per_frame(
        self,
        og_frame: Image.Image,
        og_mask: Image.Image,
        augm_frame: Image.Image,
        augm_mask: Image.Image,
        transformations_list: list,
        **kwargs,
    ) -> tuple[Image.Image]:

        resize_w, resize_h = kwargs["resize_w"], kwargs["resize_h"]

        # Reshape all
        frame_og = og_frame.resize((resize_w, resize_h))
        mask_og = og_mask.resize((resize_w, resize_h))
        frame_new = augm_frame.resize((resize_w, resize_h))
        mask_new_p = augm_mask.resize((resize_w, resize_h))

        # Apply transformations to new frame and mask
        frame_new, mask_new_p = self.apply_transformations(
            frame_new, mask_new_p, transformations_list, **kwargs
        )

        frame_augm, mask_augm = self.overlay_frames_masks(
            frame_og, mask_og, frame_new, mask_new_p
        )

        return frame_augm, mask_augm

    def augment_image(
        self,
        original_im,
        original_gt,
        new_im,
        new_gt,
        transformations_list: list[VOSTransformations],
        transformation_params: dict,
    ) -> tuple[Image.Image, Image.Image]:
        # Transforms to be applied to the new mask

        scale_factor_lower_lim = transformation_params["scale_factor_lower_lim"]
        scale_factor_upper_lim = transformation_params["scale_factor_upper_lim"]
        translation_lim = transformation_params["translation_lim"]
        resize_w = transformation_params["resize_w"]
        resize_h = transformation_params["resize_h"]

        horizontal_p = np.random.rand()

        scale_factor = np.random.uniform(
            low=scale_factor_lower_lim,
            high=scale_factor_upper_lim,
        )

        translation_w_lim, translation_h_lim = 0, 0
        if translation_lim > 0:
            translation_w_lim = np.random.randint(
                low=-int(resize_w * translation_lim),
                high=int(resize_w * translation_lim),
            )

            translation_h_lim = np.random.randint(
                low=-int(resize_h * translation_lim),
                high=int(resize_h * translation_lim),
            )

        transformation_options = {
            "resize_w": resize_w,
            "resize_h": resize_h,
            "translation": (translation_w_lim, translation_h_lim),
            "horizontal_p": horizontal_p,
            "scale_factor": scale_factor,
        }

        this_im, this_gt = self.get_augmented_data_per_frame(
            original_im,
            original_gt,
            new_im,
            new_gt,
            transformations_list,
            **transformation_options,
        )

        return this_im, this_gt
