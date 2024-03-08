from typing import Callable
from numpy.typing import ArrayLike
from torch import Tensor
import torch

from PIL import Image
import numpy as np
from pathlib import Path


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



class VOSAugmentations:

    def __init__(
        self,
        select_instances: bool,
        foreground_p: float,
        include_new_instances: bool,
        ):
        '''
        foreground_p: probability that the augmentation mask is in the foreground, occluding the original mask.
        include_new_instances: If True, the selected instances in the augmentation mask will be included in the final mask
        select_instances: if True, in the case the augmentation maks has more than one instances, all or a subset of those will
                        be included in the augmentation to be applied.
        '''
        
        
        self.select_instances = select_instances
        
        if foreground_p >1.0:
            self.foreground_p = 1.0
        elif foreground_p < 0.0:
            self.foreground_p = 0.0
        else:
            self.foreground_p = foreground_p
        
        self.include_new_instances = include_new_instances

    

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
    
    
    def select_mask_instances(self,mask_p: ArrayLike) -> ArrayLike:
        discrete_inst = np.unique(mask_p)[1:]  # the 1st element is 0
        try:
            n_inst = np.random.randint(low=1, high=len(discrete_inst) + 1)
        except ValueError:
            return mask_p
        chosen_instances = np.random.choice(discrete_inst, size=n_inst, replace=False)
        tmp_new_mask_p = np.zeros_like(mask_p)
        for i in sorted(chosen_instances):
            tmp_new_mask_p[mask_p == i] = i

        return tmp_new_mask_p
    
    
    def get_new_mask_values(self,mask_og_p: ArrayLike, mask_new_p: ArrayLike) -> ArrayLike:
        """
        Assumption: The original mask is always in ordered form
        like [0,1,2,3,4]...
        This is the case when I first open the original mask as 'P' in PIL
        """

        og_mask_vals = np.unique(mask_og_p)
        new_mask_vals = np.unique(mask_new_p)

        new_vals = np.zeros((len(og_mask_vals) + len(new_mask_vals) - 1))
        new_vals[0 : len(og_mask_vals)] = og_mask_vals

        ptr = len(og_mask_vals)
        for i in new_mask_vals[1:]:
            if i in new_vals:
                new_vals[ptr] = new_vals.max() + 1
            else:
                new_vals[ptr] = i
            ptr += 1

        # The same as the mask for augmentation, but
        # with changed values for the instances in case they were
        new_mask_copy = mask_new_p.copy()

        for i, el in enumerate(new_mask_vals[1:]):
            new_mask_copy[mask_new_p == el] = new_vals[len(og_mask_vals) + i]

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
            new_mask__ = self.get_new_mask_values(mask_og_p, mask_new_p)

        # Overlay mask of frame 2 on 1
        frame_augm = frame_og.copy()
        mask_augm = mask_og_p.copy()

        # back or foreground
        foreground = (np.random.rand() >= self.foreground_p)
        if foreground:
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
        **kwargs
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
                frame_og,
                mask_og,
                frame_new,
                mask_new_p,
                select_instances=self.select_instances,
                foreground=self.foreground,
                include_new_instances=self.include_new_instances,
            )

        # # Convert to numpy arrays
        # frame_og = np.asarray(frame_og)
        # mask_og = np.asarray(mask_og)
        # frame_new = np.asarray(frame_new)
        # mask_new_p = np.asarray(mask_new_p)

        # # Overlay mask of frame 2 on 1
        # frame_augm = frame_og.copy()
        # mask_augm = mask_og.copy()

        # frame_augm[mask_new_p > 0] = frame_new[mask_new_p > 0]
        # mask_augm[mask_new_p > 0] = 0

        # # save frames and masks
        # frame_augm = Image.fromarray(frame_augm)
        # mask_augm = Image.fromarray(mask_augm)

        return frame_augm, mask_augm
