from typing import Callable
from numpy.typing import ArrayLike
from torch import Tensor

from PIL import Image
import numpy as np
from pathlib import Path


class VOSAugmentations:
    def __init__(self) -> None:
        pass

    def random_scale(
        self, frame: ArrayLike | Tensor, mask: ArrayLike | Tensor, **kwargs
    ) -> ArrayLike | Tensor:
        """
        Resizes the image and crops the center part, sot that resulthas the same dims as input
        Scale is percentage increase or decrease: -0.99 -> 99% decrease in og dims
        """
        p = kwargs["scale_p"]
        if p < 0.5:
            return frame, mask

        s = kwargs["scale_factor"]
        if s < -0.99:
            s = -0.99

        # frame = Image.from

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

    def random_horizontal_flip(
        frame: Image.Image, mask: Image.Image, **kwargs
    ) -> tuple[Image.Image]:
        p = kwargs["horizontal_p"]
        if p >= 0.5:
            frame = frame.transpose(method=Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(method=Image.FLIP_LEFT_RIGHT)
        return frame, mask

    def random_translation(
        self, frame: Image.Image, mask: Image.Image, **kwargs
    ) -> tuple[Image.Image]:
        """
        Random translation for the frame and the mask. Padded with zeros.
        It is used as a random positioning of the new image region.
        """
        translation = kwargs["translation"]
        return frame.rotate(0, translate=translation), mask.rotate(
            0, translate=translation
        )

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
