from pathlib import Path
import random
from enum import Enum

from PIL import Image
import numpy as np


class VideoMixMode(Enum):
    Spatial = 1
    Temporal = 2
    SpatioTemporal = 3


class VideoMixDataset:
    def __init__(
        self,
        frames_path: str,
        annotations_path: str,
        validation_videos_txt_path: str | None = None,
        dataset_name: str = None,
    ):
        self.dataset_name = dataset_name
        self.video_name = None

        self.frames_path = Path(frames_path)
        self.annotations_path = Path(annotations_path)

        self.video_names = []
        if validation_videos_txt_path:
            with open(validation_videos_txt_path, "r") as f:
                val_vid_names = [i.strip() for i in f.readlines()]

            for vid in self.frames_path.iterdir():
                if vid.stem in val_vid_names:
                    continue
                self.video_names.append(vid.stem)
        else:
            self.video_names = [i.stem for i in self.frames_path.iterdir()]

    def _get_video_name(self) -> tuple[Path]:
        i = random.randint(0, len(self.video_names) - 1)
        vid_name = self.video_names[i]
        self.video_name = vid_name
        return self.frames_path.joinpath(vid_name), self.annotations_path.joinpath(
            vid_name
        )

    def get_video_frames_paths(self) -> tuple[list[Path]]:
        frames_path, annotations_path = self._get_video_name()
        frames = sorted(list(frames_path.iterdir()))
        annotations = sorted(list(annotations_path.iterdir()))

        assert set([i.stem for i in frames]) == set(
            [i.stem for i in annotations]
        ), f"Frame numbers of {frames_path.as_posix()} and masks of {annotations_path.as_posix()} do not match"

        return frames, annotations


class VideoMixer:
    def __init__(
        self,
        datasets: list[VideoMixDataset],
        video_mix_mode: VideoMixMode,
        alpha: float = 8.0,
    ):
        """
        alpha is the beta distribution hyperparameter. Default is 8.0.
        In VideoMix repo it is passed as a cli argument to the training script
        """

        # for safety
        if isinstance(datasets, VideoMixDataset):
            datasets = [datasets]

        self.datasets = datasets
        self.alpha = alpha

        self.mode = video_mix_mode

        self.selected_dataset = None

    def _rand_bbox(self, size, lam):
        W = size[0]
        H = size[1]
        # cut_rat = np.sqrt(1.0 - lam)
        cut_rat = np.sqrt(lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def _fix_mask(self, mask: np.ndarray, mask_offset: int) -> np.ndarray:
        new_mask_vals = np.unique(mask)[1:]

        for val in new_mask_vals:
            mask[mask == val] = val + mask_offset

        return mask

    def videomix_algorithm(self, t: int, h: int, w: int) -> tuple[int]:
        """
        t: number of frames. Adjusted so that both sequences have equal length
        h,w: height and width of the sequences. The augmentation has been resized to the original's dims
        """

        lambda_ = np.random.uniform(0, 1)
        # check mode
        if self.mode == VideoMixMode.Spatial:
            # tubemix-spatial

            # lambda_ = np.random.beta(self.alpha, self.alpha)
            wmin, hmin, wmax, hmax = self._rand_bbox((w, h), lambda_)

            tmin = 0
            tmax = t

        elif self.mode == VideoMixMode.Temporal:
            # stackmix
            wmin, hmin, wmax, hmax = 0, 0, w, h
            tc = np.random.randint(t)

            t_cut = int(lambda_ * t)

            tmin = np.clip(tc - t_cut // 2, 0, t)
            tmax = np.clip(tc + t_cut // 2, 0, t)

        else:
            # spatiotemporal-both
            wc, hc, tc = (
                np.random.randint(w),
                np.random.randint(h),
                np.random.randint(t),
            )

            mlt = np.power(lambda_, 1 / 3)

            t_cut = int(mlt * t)
            h_cut = int(mlt * h)
            w_cut = int(mlt * w)

            tmin = np.clip(tc - t_cut // 2, 0, t)
            tmax = np.clip(tc + t_cut // 2, 0, t)

            wmin = np.clip(wc - w_cut // 2, 0, w)
            wmax = np.clip(wc + w_cut // 2, 0, w)
            hmin = np.clip(hc - h_cut // 2, 0, h)
            hmax = np.clip(hc + h_cut // 2, 0, h)

        return tmin, tmax, hmin, hmax, wmin, wmax

    def videomix(
        self, original_frame_paths: list[str], original_masks_path: list[str]
    ) -> list[Image.Image]:

        assert len(original_frame_paths) == len(
            original_masks_path
        ), "Original frames and annotations directories do not contain the same number of files"

        video_mix_dataset_i: int = random.randint(0, len(self.datasets) - 1)
        video_mix_dataset: VideoMixDataset = self.datasets[video_mix_dataset_i]

        self.selected_dataset = video_mix_dataset  # for debugging

        augmentation_frames_paths, augmentation_annots_paths = (
            video_mix_dataset.get_video_frames_paths()
        )

        # to resize new frame-mask pairs, because different datasets are used, eg.DAVIS, YT-VOS
        original_annot_tmp = Image.open(original_masks_path[0]).convert("P")
        w, h = original_annot_tmp.size

        # to change augmentation mask palette colour map, add an offset to all augmentation mask values
        original_class_vals = np.unique(np.array(original_annot_tmp))
        mask_offset = original_class_vals[-1]
        if mask_offset == 255 and len(original_class_vals) > 2:
            mask_offset = original_class_vals[-2]

        # durations of the two videos
        t1 = len(original_frame_paths)
        t2 = len(augmentation_frames_paths)
        # In order to replicate, both videos should
        # have the same length
        t = min(t1, t2)

        tmin, tmax, hmin, hmax, wmin, wmax = self.videomix_algorithm(t, h, w)

        # here I will have to open all images in lists and make them numpy 'tensors'
        # this is the worst part memory-wise. it is not on gpu, but still...
        original_frames = []
        original_masks = []
        for frame, mask in zip(original_frame_paths, original_masks_path):
            original_frames.append(np.array(Image.open(frame).convert("RGB")))
            original_masks.append(np.array(Image.open(mask).convert("P")))

        original_frames = np.array(original_frames)
        original_masks = np.array(original_masks)

        assert original_frames.shape == (
            t1,
            h,
            w,
            original_frames[0].shape[-1],
        ), "Conversion to numpy tensor did not succeed"

        # same for augmentation, but now I have to resize to original's dims
        augmentation_frames = []
        augmentation_masks = []
        for frame, mask in zip(augmentation_frames_paths, augmentation_annots_paths):
            tmp_frame = Image.open(frame).convert("RGB").resize((w, h))
            augmentation_frames.append(np.array(tmp_frame))

            tmp_mask = np.array(Image.open(mask).convert("P").resize((w, h)))
            tmp_mask = self._fix_mask(tmp_mask, mask_offset)
            augmentation_masks.append(tmp_mask)

        augmentation_frames = np.array(augmentation_frames)
        augmentation_masks = np.array(augmentation_masks)

        # now combine
        original_frames[tmin:tmax, hmin:hmax, wmin:wmax, :] = augmentation_frames[
            tmin:tmax, hmin:hmax, wmin:wmax, :
        ]
        original_masks[tmin:tmax, hmin:hmax, wmin:wmax] = augmentation_masks[
            tmin:tmax, hmin:hmax, wmin:wmax
        ]

        if self.mode == VideoMixMode.Spatial:
            original_frames = original_frames[:t]
            original_masks = original_masks[:t]

        # return PIL.Image arrays
        final_frames = []
        final_masks = []
        for i in range(original_frames.shape[0]):
            frame_ = Image.fromarray(original_frames[i])
            mask_ = Image.fromarray(original_masks[i])

            final_frames.append(frame_)
            final_masks.append(mask_)
        return final_frames, final_masks
