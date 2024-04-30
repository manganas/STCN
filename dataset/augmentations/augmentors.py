from pathlib import Path

import numpy as np
from PIL import Image

from dataset.data_augmentations_static_motion import StaticImagesAugmentor

from dataset.data_augmentations_coco import CocoAugmentor
from pycocotools.coco import COCO


class DAVISAugmentor:
    def __init__(self, root_path, max_jump: int = 5):

        self.max_jump = max_jump

        self.im_path = Path(root_path).joinpath("JPEGImages", "480p")
        self.gt_path = Path(root_path).joinpath("Annotations", "480p")

        imagesets = Path(root_path).joinpath("ImageSets", "2017", "train.txt")

        # Keep only videos from training set
        with open(imagesets, "r") as f:
            vid_names = f.read().strip().split("\n")

        all_videos = list(self.im_path.iterdir())
        all_video_names = [name.stem for name in all_videos]

        self.videos = []
        for vid in all_video_names:
            if vid not in vid_names:
                continue
            self.videos.append(vid)

    def __len__(self) -> int:
        return len(self.videos)

    def get_vid_frames_paths(
        self, idx: int
    ) -> tuple[list[Path], list[Path], list[str]]:
        video = self.videos[idx]
        vid_im_path = self.im_path.joinpath(video)
        vid_gt_path = self.gt_path.joinpath(video)
        frames = sorted(list(vid_im_path.iterdir()))
        frames = [frame.stem for frame in frames]
        return vid_im_path, vid_gt_path, frames

    def get_frames_indices(self, frames: list) -> list[int]:
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

        return frames_idx

    def get_augmentation_data(self) -> tuple[list[Path], list[Path]]:

        # Get a random data point
        j = np.random.randint(low=0, high=self.__len__())
        vid_im_path, vid_gt_path, frames = self.get_vid_frames_paths(j)
        frames_idx = self.get_frames_indices(frames)  # includes reversal

        images_paths = []
        masks_paths = []

        for idx in frames_idx:
            # frames
            tmp_path = vid_im_path.joinpath(frames[idx].zfill(5) + ".jpg")
            images_paths.append(tmp_path)

            # masks
            tmp_path = vid_gt_path.joinpath(frames[idx].zfill(5) + ".png")
            masks_paths.append(tmp_path)

        frames = []
        masks = []

        for i in range(len(images_paths)):
            frames.append(Image.open(images_paths[i]).convert("RGB"))
            masks.append(Image.open(masks_paths[i]).convert("P"))

        return frames, masks


class YTAugmentor:
    def __init__(self, root_path, max_jump: int = 1):

        self.im_path = Path(root_path).joinpath("JPEGImages")
        self.gt_path = Path(root_path).joinpath("Annotations")
        self.max_jump = max_jump

        # remove validation videos, keep  training only
        self.videos = []
        training_videos_yt_path = Path("./util/yv_subset.txt")
        with open(training_videos_yt_path, "r") as f:
            vid_names = f.read().strip().split("\n")

        all_videos = list(self.im_path.iterdir())
        all_video_names = [name.stem for name in all_videos]

        self.videos = []
        for vid in all_video_names:
            if vid not in vid_names:
                continue
            self.videos.append(vid)

    def __len__(self) -> int:
        return len(self.videos)

    def get_vid_frames_paths(
        self, idx: int
    ) -> tuple[list[Path], list[Path], list[str]]:
        video = self.videos[idx]
        vid_im_path = self.im_path.joinpath(video)
        vid_gt_path = self.gt_path.joinpath(video)
        frames = sorted(list(vid_im_path.iterdir()))
        frames = [frame.stem for frame in frames]
        return vid_im_path, vid_gt_path, frames

    def get_frames_indices(self, frames: list) -> list[int]:
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

        return frames_idx

    def get_augmentation_data(self) -> tuple[list[Path], list[Path]]:

        # Get a random data point
        j = np.random.randint(low=0, high=self.__len__())
        vid_im_path, vid_gt_path, frames = self.get_vid_frames_paths(j)
        frames_idx = self.get_frames_indices(frames)  # includes reversal

        images_paths = []
        masks_paths = []

        for idx in frames_idx:
            # frames
            tmp_path = vid_im_path.joinpath(frames[idx].zfill(5) + ".jpg")
            images_paths.append(tmp_path)

            # masks
            tmp_path = vid_gt_path.joinpath(frames[idx].zfill(5) + ".png")
            masks_paths.append(tmp_path)

        frames = []
        masks = []

        for i in range(len(images_paths)):
            frames.append(Image.open(images_paths[i]).convert("RGB"))
            masks.append(Image.open(masks_paths[i]).convert("P"))

        return frames, masks


class COCOAugmentor:
    def __init__(self, coco_root: Path, davis_root: Path):

        self.max_jump = 5

        self.davis_root = davis_root
        self.coco_augmentor = CocoAugmentor(coco_root, davis_root)

        # COCO related
        coco_annotations_path = coco_root.joinpath(
            "annotations", "instances_train2017.json"
        )
        self.coco_images_path = coco_root.joinpath("train2017")
        self.coco = COCO(coco_annotations_path)

        self.img_Ids = sorted(self.coco.getImgIds())

        self.davis_videos = self.coco_augmentor.davis_video_names

    def __len__(self):
        return len(self.davis_videos)

    def get_frames_indices(self, frames: list) -> list[int]:
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

        return frames_idx

    def get_augmentation_data(self) -> tuple[list[Path], list[Path]]:

        # Get a random data point
        j = np.random.randint(low=0, high=self.__len__())

        coco_frames_list, coco_masks_list = self.coco_augmentor.get_augmentation_data(
            self.davis_videos[j]
        )
        frames_idx = self.get_frames_indices(coco_frames_list)  # includes reversal

        frames = []
        masks = []

        for idx in frames_idx:
            frames.append(coco_frames_list[idx])
            masks.append(coco_masks_list[idx])

        return frames, masks


class StaticAugmentor:
    def __init__(self, root_dir: Path, davis_root: Path):
        # needs glob to find jpg's and then pair it with png
        self.root_dir = root_dir

        self.max_jump = 5

        # Use all images
        self.all_images = sorted(list(root_dir.glob("*.jpg")))
        self.all_masks = sorted(list(root_dir.glob("*.png")))

        assert len(self.all_images) == len(
            self.all_masks
        ), f"Not the same number of images and masks for dataset {root_dir.as_posix()}"

        self.static_frame_augmentor = StaticImagesAugmentor(root_dir, davis_root)
        self.davis_videos = self.static_frame_augmentor.davis_video_names

    def __len__(self):
        return len(self.static_frame_augmentor.davis_video_names)

    def get_frames_indices(self, frames: list) -> list[int]:
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

        return frames_idx

    def get_augmentation_data(self) -> tuple[list[Path], list[Path]]:

        # Get a random data point
        j = np.random.randint(low=0, high=self.__len__())

        coco_frames_list, coco_masks_list = (
            self.static_frame_augmentor.get_augmentation_data(self.davis_videos[j])
        )
        frames_idx = self.get_frames_indices(coco_frames_list)  # includes reversal

        frames = []
        masks = []

        for idx in frames_idx:
            frames.append(coco_frames_list[idx])
            masks.append(coco_masks_list[idx])

        return frames, masks


class FSSAugmentor:
    def __init__(self, root_dir: Path, davis_root: Path):
        self.root_dir = root_dir

        self.max_jump = 5

        # Use all images
        self.all_images = sorted(list(root_dir.glob("*.jpg")))
        self.all_masks = sorted(list(root_dir.glob("*.png")))

        assert len(self.all_images) == len(
            self.all_masks
        ), f"Not the same number of images and masks for dataset {root_dir.as_posix()}"

        self.static_frame_augmentor = StaticImagesAugmentor(root_dir, davis_root)
        self.davis_videos = self.static_frame_augmentor.davis_video_names

    def __len__(self):
        return len(self.static_frame_augmentor.davis_video_names)

    def get_frames_indices(self, frames: list) -> list[int]:
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

        return frames_idx

    def get_augmentation_data(self) -> tuple[list[Path], list[Path]]:

        # Get a random data point
        j = np.random.randint(low=0, high=self.__len__())

        coco_frames_list, coco_masks_list = (
            self.static_frame_augmentor.get_augmentation_data(self.davis_videos[j])
        )
        frames_idx = self.get_frames_indices(coco_frames_list)  # includes reversal

        frames = []
        masks = []

        for idx in frames_idx:
            frames.append(coco_frames_list[idx])
            masks.append(coco_masks_list[idx])

        return frames, masks
