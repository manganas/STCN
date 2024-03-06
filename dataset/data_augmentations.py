from typing import Callable
from numpy.typing import ArrayLike
from torch import Tensor
import torch

from PIL import Image
import numpy as np
from pathlib import Path


class VOSTransformations:

    def random_scale(p_scale: float):
        def random_scale_wrapper(
            frame: ArrayLike | Tensor, mask: ArrayLike | Tensor, **kwargs
        ) -> ArrayLike | Tensor:
            """
            Resizes the image and crops the center part, sot that resulthas the same dims as input
            Scale is percentage increase or decrease: -0.99 -> 99% decrease in original dims
            """
            p = kwargs["scale_p"]
            if p >= p_scale:
                # print("No scaling")
                return frame, mask

            # print("Scaling")
            s = kwargs["scale_factor"]
            if s < -0.99:
                s = -0.99

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

        return random_scale_wrapper

    def random_horizontal_flip(p_horizontal_flip: float):
        def random_horizontal_flip_wrapper(
            frame: Image.Image, mask: Image.Image, **kwargs
        ) -> tuple[Image.Image]:

            p = kwargs["horizontal_p"]
            if p >= p_horizontal_flip:
                # print("No flipping")
                return frame, mask

            # print("Flipping")
            frame = frame.transpose(method=Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(method=Image.FLIP_LEFT_RIGHT)
            return frame, mask

        return random_horizontal_flip_wrapper

    def random_translation(p_translation: float):
        def random_translation_wrapper(
            frame: Image.Image, mask: Image.Image, **kwargs
        ) -> tuple[Image.Image]:
            """
            Random translation for the frame and the mask. Padded with zeros.
            It is used as a random positioning of the new image region.
            """

            p = kwargs["translation_p"]

            if p >= p_translation:
                # print("No translation")
                return frame, mask

            # print("Translation")
            translation = kwargs["translation"]

            return frame.rotate(0, translate=translation), mask.rotate(
                0, translate=translation
            )

        return random_translation_wrapper


class VOSAugmentations:

    def __init__(
        self,
        select_instances: bool,
        always_foreground: bool,
        include_new_instances: bool,
    ):
        self.select_instances = select_instances
        self.always_foreground = always_foreground
        self.include_new_instances = include_new_instances

    # def random_scale(
    #     frame: ArrayLike | Tensor, mask: ArrayLike | Tensor, **kwargs
    # ) -> ArrayLike | Tensor:
    #     """
    #     Resizes the image and crops the center part, sot that resulthas the same dims as input
    #     Scale is percentage increase or decrease: -0.99 -> 99% decrease in og dims
    #     """
    #     p = kwargs["scale_p"]
    #     if p < 0.5:
    #         return frame, mask

    #     s = kwargs["scale_factor"]
    #     if s < -0.99:
    #         s = -0.99

    #     # frame = Image.from

    #     w, h = frame.size
    #     w_s = int((1 + s) * w)
    #     h_s = int((1 + s) * h)

    #     frame = frame.resize((w_s, h_s))
    #     mask = mask.resize((w_s, h_s))

    #     w_mid = w_s // 2
    #     h_mid = h_s // 2
    #     left, top, right, bottom = (
    #         w_mid - w // 2,
    #         h_mid - h // 2,
    #         w_mid + w // 2,
    #         h_mid + h // 2,
    #     )
    #     frame = frame.crop((left, top, right, bottom))
    #     mask = mask.crop((left, top, right, bottom))

    #     return frame, mask

    # def random_horizontal_flip(
    #     frame: Image.Image, mask: Image.Image, **kwargs
    # ) -> tuple[Image.Image]:
    #     p = kwargs["horizontal_p"]
    #     if p >= 0.5:
    #         frame = frame.transpose(method=Image.FLIP_LEFT_RIGHT)
    #         mask = mask.transpose(method=Image.FLIP_LEFT_RIGHT)
    #     return frame, mask

    # def random_translation(
    #     frame: Image.Image, mask: Image.Image, **kwargs
    # ) -> tuple[Image.Image]:
    #     """
    #     Random translation for the frame and the mask. Padded with zeros.
    #     It is used as a random positioning of the new image region.
    #     """
    #     translation = kwargs["translation"]
    #     return frame.rotate(0, translate=translation), mask.rotate(
    #         0, translate=translation
    #     )

    def apply_transformations(
        self,
        frame: Image.Image,
        mask: Image.Image,
        transformations_list: list = [],
        **kwargs
    ) -> tuple[Image.Image]:
        for transform in transformations_list:
            frame, mask = transform(frame, mask, **kwargs)

        return frame, mask

    def get_augmented_data_per_frame(
        self,
        og_frame: Image.Image,
        og_mask: Image.Image,
        augm_frame: Image.Image,
        augm_mask: Image.Image,
        transformations_list: list,
        **kwargs
    ) -> tuple[Image.Image]:

        resize_w, resize_h = kwargs["resize_w"], kwargs["resize_h"]

        # Reshape all
        frame_og = og_frame.resize((resize_w, resize_h))
        mask_og = og_mask.resize((resize_w, resize_h))
        frame_new = augm_frame.resize((resize_w, resize_h))
        # mask_new = mask_new.resize((resize_w, resize_h))
        mask_new_p = augm_mask.resize((resize_w, resize_h))

        # Apply transformations to new frame and mask
        frame_new, mask_new_p = self.apply_transformations(
            frame_new, mask_new_p, transformations_list, **kwargs
        )

        # Convert to numpy arrays
        frame_og = np.asarray(frame_og)
        mask_og = np.asarray(mask_og)
        frame_new = np.asarray(frame_new)
        mask_new_p = np.asarray(mask_new_p)

        # Overlay mask of frame 2 on 1
        frame_augm = frame_og.copy()
        mask_augm = mask_og.copy()

        frame_augm[mask_new_p > 0] = frame_new[mask_new_p > 0]
        mask_augm[mask_new_p > 0] = 0

        # save frames and masks
        frame_augm = Image.fromarray(frame_augm)
        mask_augm = Image.fromarray(mask_augm)

        return frame_augm, mask_augm
