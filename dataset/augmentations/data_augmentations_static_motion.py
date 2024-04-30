from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from PIL import Image

from dataset.augmentations.bounding_boxes import (
    get_bbox_start_end,
    calculate_bbox_center,
    convert_bbox_dims_to_width,
    get_bbox_box_width,
)


def rescale_mask(
    prototype_mask: NDArray,
    to_transform_mask: NDArray,
    to_transform_frame: NDArray | None = None,
) -> tuple[NDArray | None]:
    """
    Rescale new mask so that the new mask has the same dims as og mask.
    Keep aspect ratio of new mask.
    Then crop to og mask dimensions
    """

    # original mask data
    h, w = prototype_mask.shape[:2]
    _, _, bbox_h_1, bbox_w_1 = get_bbox_box_width(np.array(prototype_mask))

    # resize new mask
    rmin, cmin, rmax, cmax = get_bbox_start_end(to_transform_mask)
    new_mask = Image.fromarray(to_transform_mask)
    new_mask = np.array(new_mask.crop((cmin, rmin, cmax, rmax)))

    scale = min(bbox_h_1 / abs(rmax - rmin), bbox_w_1 / abs(cmax - cmin))

    resize_h, resize_w = int(scale * new_mask.shape[0]), int(scale * new_mask.shape[1])

    new_mask = np.array(Image.fromarray(new_mask).resize((resize_w, resize_h)))

    new_mask_tmp = np.zeros_like(prototype_mask)

    row_start, column_start = np.random.randint(
        low=0, high=h - new_mask.shape[0]
    ), np.random.randint(low=0, high=w - new_mask.shape[1])

    new_mask_tmp[
        row_start : row_start + new_mask.shape[0],
        column_start : column_start + new_mask.shape[1],
    ] = new_mask

    if to_transform_frame is not None:
        new_frame = np.array(
            Image.fromarray(to_transform_frame)
            .crop((cmin, rmin, cmax, rmax))
            .resize((resize_w, resize_h))
        )

        new_frame_tmp = np.zeros(
            prototype_mask.shape + (to_transform_frame.shape[-1],), dtype=np.uint8
        )

        # for i in range(new_frame_tmp.shape[-1]):
        new_frame_tmp[
            row_start : row_start + new_frame.shape[0],
            column_start : column_start + new_frame.shape[1],
        ] = new_frame

        return new_mask_tmp, new_frame_tmp

    return new_mask_tmp, None


def select_instance(mask: NDArray, instance: int) -> NDArray:
    new_mask = np.zeros_like(mask).astype(np.uint8)
    new_mask[mask == instance] = 1
    return new_mask


class StaticImagesAugmentor:
    def __init__(self, static_dataset_root: Path, davis_root_path: Path):

        # DAVIS related
        davis_root_path = Path(davis_root_path)
        self.davis_frames = davis_root_path.joinpath("JPEGImages", "480p")
        self.davis_annotations = davis_root_path.joinpath("Annotations", "480p")
        davis_video_names_ = [vid_name.stem for vid_name in self.davis_frames.iterdir()]

        imagesets = davis_root_path.joinpath("ImageSets", "2017", "train.txt")

        # Keep only videos from training set
        try:
            with open(imagesets, "r") as f:
                vid_names = f.read().strip().split("\n")

            self.davis_video_names = []
            for vid in davis_video_names_:
                if vid not in vid_names:
                    continue
                self.davis_video_names.append(vid)
        except FileNotFoundError as e:
            print(e)
            self.davis_video_names = davis_video_names_

        # Use all images
        self.static_dataset_root = Path(static_dataset_root)
        self.all_images = sorted(list(static_dataset_root.glob("**/*.jpg")))
        self.all_masks = sorted(list(static_dataset_root.glob("**/*.png")))

        assert len(self.all_images) == len(
            self.all_masks
        ), f"Not the same number of images and masks for dataset {static_dataset_root.as_posix()}"

    def remove_border_objects(
        self,
        mask: NDArray,
        px_padding=15,
    ) -> NDArray:
        """
        Prepare the mask, so that no mask is selected with borders on the
        edge of the image shape.
        The davis shape is used so that the mask can be positioned and scaled
        relative to the davis image.
        """

        mask_h, mask_w = mask.shape[:2]
        instances = np.unique(mask)[1:]

        selected_instances = []

        for instance in instances:
            rows, cols = np.where(mask == instance)

            if len(rows) == 0 or len(cols) == 0:
                continue

            if (
                rows.min() <= px_padding
                or cols.min() <= px_padding
                or rows.max() >= mask_h - px_padding
                or cols.max() >= mask_w - px_padding
            ):
                continue

            selected_instances.append(instance)

        new_mask = np.zeros_like(mask)
        for instance in selected_instances:
            new_mask[mask == instance] = instance

        return new_mask

    def _get_static_data(
        self,
        resize_shape_h_w: tuple[int] | None = None,
        low_threshold: float = 0.05,
        high_threshold: float = 0.5,
    ) -> tuple[NDArray]:

        found = False

        while not found:
            img_idx = np.random.randint(low=0, high=len(self.all_images))

            mask_tmp = self.all_masks[img_idx]
            mask_tmp = np.array(Image.open(mask_tmp).convert("P"))

            # care needed if I know I have multiple objects and
            # I want to isolate one
            mask_tmp[mask_tmp > 0] = 1

            img_area = mask_tmp.shape[0] * mask_tmp.shape[1]

            objects = np.unique(mask_tmp)[1:]

            if len(objects) == 0:
                continue

            mask_tmp = self.remove_border_objects(mask_tmp, px_padding=25)
            objects = np.unique(mask_tmp)[1:]

            if len(objects) == 0:
                continue

            min_dist = 100000
            min_i = -1

            # mask = mask_tmp

            for jj, object in enumerate(objects):

                mask = np.zeros_like(mask_tmp)
                mask[mask_tmp == object] = 1

                bbox_dims = get_bbox_start_end(mask)
                bbox_center = calculate_bbox_center(bbox_dims)
                bbox_dist = np.sqrt(bbox_center[0] ** 2 + bbox_center[1] ** 2)

                if bbox_dist < min_dist:
                    min_dist = bbox_dist
                    min_i = jj

            if min_i == -1:
                continue

            mask = np.zeros_like(mask_tmp)
            mask[mask_tmp == objects[min_i]] = 1

            if mask.sum() == 0:
                print("Zero mask returned!")

                # # possible inf loop!
                # continue

            break

            # if (
            #     mask.sum() >= low_threshold * img_area
            #     and mask.sum() < high_threshold * img_area
            # ):

            #     found = True
            #     break

        img_path = self.all_images[img_idx]
        img = np.array(Image.open(img_path).convert("RGB"))

        if resize_shape_h_w:
            mask = Image.fromarray(mask).resize(resize_shape_h_w[::-1])
            img = Image.fromarray(img).resize(resize_shape_h_w[::-1])

            img = np.array(img)
            mask = np.array(mask)

        return img, mask

    def _get_davis_video_name_instance(
        self, vid_name: int, low_px_threshold: int = 100
    ) -> tuple[str, int]:

        # vid_name = self.davis_video_names[idx]
        mask = self.davis_annotations.joinpath(vid_name, "00000.png")
        mask = np.array(Image.open(mask).convert("P"))
        permuted_instances = np.unique(mask)[1:]

        # permuted_instances = np.random.permutation(np.unique(mask)[1:])

        for i in permuted_instances:
            if mask[mask == i].sum() >= low_px_threshold:
                break

        return vid_name, i

    def _sample_davis_video(self, davis_vid_name: str, instance: int) -> tuple[NDArray]:
        # Initialize scale, starting point

        video_path = self.davis_annotations.joinpath(davis_vid_name)

        mask = video_path.joinpath("00000.png")
        mask = np.array(Image.open(mask).convert("P"))
        h, w = mask.shape
        bbox_0 = convert_bbox_dims_to_width(get_bbox_start_end(mask))
        bbox_0 = np.array(bbox_0)

        frames = sorted(video_path.iterdir())
        scale_h = np.zeros(len(frames))
        scale_w = np.zeros(len(frames))
        row_offset = np.zeros(len(frames))
        col_offset = np.zeros(len(frames))

        for ii, frame in enumerate(frames):
            # get bbox of frame
            mask = np.array(Image.open(frame).convert("P"))
            mask = select_instance(mask, instance)
            try:
                bbox_i = convert_bbox_dims_to_width(get_bbox_start_end(mask))
                bbox_i = np.array(bbox_i)
                offsets = bbox_i[:2] - bbox_0[:2]
                scales = bbox_i[2:] / bbox_0[2:]

                scale_h[ii] = scales[0]
                scale_w[ii] = scales[1]

                row_offset[ii] = offsets[0]
                col_offset[ii] = offsets[1]
            except IndexError:
                # lost mask. do not move in this case.
                # can also reverse
                scale_h[ii] = 1  # scales[ii - 1] if ii > 0 else 1
                scale_w[ii] = 1  # scales[ii - 1] if ii > 0 else 1

                row_offset[ii] = 0 if ii == 0 else row_offset[ii - 1]
                col_offset[ii] = 0 if ii == 0 else col_offset[ii - 1]

        offsets_scales = (
            row_offset,
            col_offset,
            scale_h,
            scale_w,
        )

        return offsets_scales, (h, w)

    def _sample_davis_video_frame(
        self, davis_vid_name: str, instance: int, frame_number: int
    ) -> tuple[NDArray]:
        # Initialize scale, starting point

        video_path = self.davis_annotations.joinpath(davis_vid_name)

        mask = video_path.joinpath("00000.png")
        mask = np.array(Image.open(mask).convert("P"))
        bbox_0 = convert_bbox_dims_to_width(get_bbox_start_end(mask))
        bbox_0 = np.array(bbox_0)

        frames = sorted(video_path.iterdir())

        frame = frames[frame_number]

        # get bbox of frame
        mask = np.array(Image.open(frame).convert("P"))
        mask = select_instance(mask, instance)
        bbox_i = convert_bbox_dims_to_width(get_bbox_start_end(mask))
        bbox_i = np.array(bbox_i)
        offsets = bbox_i[:2] - bbox_0[:2]
        scales = bbox_i[2:] / bbox_0[2:]

        offsets_scales = offsets[0], offsets[1], scales[0], scales[1]
        return offsets_scales

    def generate_fake_frames_masks(
        self,
        static_frame: NDArray,
        static_mask: NDArray,
        offsets_scales: tuple[NDArray],
    ) -> tuple[list[Image.Image]]:
        static_frames = []
        static_masks = []

        static_frame = Image.fromarray(static_frame)
        static_mask = Image.fromarray(static_mask)
        static_frame = np.array(static_frame)
        static_mask = np.array(static_mask)

        offset_row, offset_col, scale_h, scale_w = offsets_scales

        for i in range(len(offsets_scales[0])):

            if (
                offset_col[i] < (static_mask.shape[1] // 2 + 10)
                or offset_col[i] >= static_mask.shape[1] // 2 - 10
            ):
                offset_col[i] = -offset_col[i]

            if (
                offset_row[i] < (static_mask.shape[0] // 2 + 10)
                or offset_row[i] >= static_mask.shape[0] // 2 - 10
            ):
                offset_row[i] = -offset_row[i]

            affine_transf = (
                1,  # * scale_w[i],
                0,
                -offset_col[i] - 0 * static_frame.shape[1],
                0,
                1,  # * scale_h[i],
                -offset_row[i] - 0 * static_frame.shape[0],
            )
            new_mask = Image.fromarray(static_mask)
            new_mask = new_mask.transform(
                new_mask.size, Image.AFFINE, data=affine_transf, fill=0
            )
            new_mask = np.array(new_mask)

            new_frame = Image.fromarray(static_frame)
            new_frame = new_frame.transform(
                new_frame.size, Image.AFFINE, data=affine_transf, fill=0
            )
            new_frame = np.array(new_frame)
            new_frame[new_mask <= 0] = 0

            static_masks.append(Image.fromarray(new_mask))
            static_frames.append(Image.fromarray(new_frame))

        return static_frames, static_masks

    def get_augmentation_data(self, davis_video_name: str) -> tuple[list[Image.Image]]:

        davis_vid_name, instance = self._get_davis_video_name_instance(davis_video_name)
        offsets_scales, resize_shape_h_w = self._sample_davis_video(
            davis_vid_name, instance
        )
        static_frame, static_mask = self._get_static_data(resize_shape_h_w)

        # rescale the object so that the relative size distribution is similar
        # to davis

        davis_mask = Image.open(
            self.davis_annotations.joinpath(davis_vid_name, "00000.png")
        ).convert("P")
        davis_mask = np.asarray(davis_mask)
        davis_mask_tmp = np.zeros_like(davis_mask)
        davis_mask_tmp[davis_mask == instance] = 1

        static_mask, static_frame = rescale_mask(
            davis_mask_tmp, static_mask, static_frame
        )

        # generate fake motion
        static_frames, static_masks = self.generate_fake_frames_masks(
            static_frame, static_mask, offsets_scales
        )

        return static_frames, static_masks


############
############
############
############
############


class CocoAugmentor:

    def __init__(self, coco_root_dir: Path, davis_root_path: Path):
        from pycocotools.coco import COCO

        coco_root_dir = Path(coco_root_dir)
        davis_root_path = Path(davis_root_path)

        # DAVIS related
        self.davis_frames = davis_root_path.joinpath("JPEGImages", "480p")
        self.davis_annotations = davis_root_path.joinpath("Annotations", "480p")
        davis_video_names_ = [vid_name.stem for vid_name in self.davis_frames.iterdir()]

        imagesets = davis_root_path.joinpath("ImageSets", "2017", "train.txt")

        # Keep only videos from training set
        try:
            with open(imagesets, "r") as f:
                vid_names = f.read().strip().split("\n")

            self.davis_video_names = []
            for vid in davis_video_names_:
                if vid not in vid_names:
                    continue
                self.davis_video_names.append(vid)
        except FileNotFoundError as e:
            print(e)
            self.davis_video_names = davis_video_names_

        # COCO related
        coco_annotations_path = coco_root_dir.joinpath(
            "annotations", "instances_train2017.json"
        )
        self.coco_images_path = coco_root_dir.joinpath("train2017")
        self.coco = COCO(coco_annotations_path)

        self.img_Ids = sorted(self.coco.getImgIds())

    def prepare_coco_mask(
        self,
        coco_mask: NDArray,
        px_padding=15,
    ) -> NDArray:
        """
        Prepare the coco mask, so that no mask is selected with borders on the
        edge of the image shape.
        The davis shape is used so that the coco mask can be positioned and scaled
        relative to the davis image.
        """

        coco_h, coco_w = coco_mask.shape[:2]
        instances = np.unique(coco_mask)[1:]
        selected_instances = []

        for instance in instances:
            rows, cols = np.where(coco_mask == instance)

            if (
                rows.min() <= px_padding
                or cols.min() <= px_padding
                or rows.max() >= coco_h - px_padding
                or cols.max() >= coco_w - px_padding
            ):
                continue

            selected_instances.append(instance)

        new_mask = np.zeros_like(coco_mask)
        for instance in selected_instances:
            new_mask[coco_mask == instance] = instance

        return new_mask

    def _get_coco_data(
        self,
        resize_shape_h_w: tuple[int] | None = None,
        low_threshold: float = 0.05,
        high_threshold: float = 0.5,
    ) -> tuple[NDArray]:

        found = False
        while not found:
            img_idx = np.random.choice(self.img_Ids, size=1)
            img_dict = self.coco.loadImgs(img_idx)[0]

            annIds = self.coco.getAnnIds(imgIds=img_dict["id"], iscrowd=None)
            anns = self.coco.loadAnns(annIds)

            img_area = img_dict["height"] * img_dict["width"]

            permuted_ans = np.random.permutation(anns)

            min_dist = 10000
            min_i = -1

            for jj, ann in enumerate(permuted_ans):
                mask = self.prepare_coco_mask(self.coco.annToMask(ann), px_padding=45)

                if len(np.unique(mask)[1:]) == 0:
                    continue

                # if ann["iscrowd"]:
                #     continue

                bbox_dims = get_bbox_start_end(mask)
                bbox_center = calculate_bbox_center(bbox_dims)
                bbox_dist = np.sqrt(bbox_center[0] ** 2 + bbox_center[1] ** 2)

                if bbox_dist < min_dist:
                    min_dist = bbox_dist
                    min_i = jj

            if min_i == -1:
                continue

            mask = self.coco.annToMask(anns[min_i])
            if (
                mask.sum() >= low_threshold * img_area
                and mask.sum() < high_threshold * img_area
            ):

                found = True
                break

        img_path = self.coco_images_path.joinpath(img_dict["file_name"])
        img = np.array(Image.open(img_path).convert("RGB"))

        if resize_shape_h_w:
            mask = Image.fromarray(mask).resize(resize_shape_h_w[::-1])
            img = Image.fromarray(img).resize(resize_shape_h_w[::-1])

            img = np.array(img)
            mask = np.array(mask)

        return img, mask

    def _get_davis_video_name_instance(
        self, vid_name: int, low_px_threshold: int = 100
    ) -> tuple[str, int]:

        # vid_name = self.davis_video_names[idx]
        mask = self.davis_annotations.joinpath(vid_name, "00000.png")
        mask = np.array(Image.open(mask).convert("P"))
        permuted_instances = np.unique(mask)[1:]

        # permuted_instances = np.random.permutation(np.unique(mask)[1:])

        for i in permuted_instances:
            if mask[mask == i].sum() >= low_px_threshold:
                break

        return vid_name, i

    def _sample_davis_video(self, davis_vid_name: str, instance: int) -> tuple[NDArray]:
        # Initialize scale, starting point

        video_path = self.davis_annotations.joinpath(davis_vid_name)

        mask = video_path.joinpath("00000.png")
        mask = np.array(Image.open(mask).convert("P"))
        h, w = mask.shape
        bbox_0 = convert_bbox_dims_to_width(get_bbox_start_end(mask))
        bbox_0 = np.array(bbox_0)

        frames = sorted(video_path.iterdir())
        scale_h = np.zeros(len(frames))
        scale_w = np.zeros(len(frames))
        row_offset = np.zeros(len(frames))
        col_offset = np.zeros(len(frames))

        for ii, frame in enumerate(frames):
            # get bbox of frame
            mask = np.array(Image.open(frame).convert("P"))
            mask = select_instance(mask, instance)
            try:
                bbox_i = convert_bbox_dims_to_width(get_bbox_start_end(mask))
                bbox_i = np.array(bbox_i)
                offsets = bbox_i[:2] - bbox_0[:2]
                scales = bbox_i[2:] / bbox_0[2:]

                scale_h[ii] = scales[0]
                scale_w[ii] = scales[1]

                row_offset[ii] = offsets[0]
                col_offset[ii] = offsets[1]
            except IndexError:
                # lost mask. do not move in this case.
                # can also reverse
                scale_h[ii] = 1  # scales[ii - 1] if ii > 0 else 1
                scale_w[ii] = 1  # scales[ii - 1] if ii > 0 else 1

                row_offset[ii] = 0 if ii == 0 else row_offset[ii - 1]
                col_offset[ii] = 0 if ii == 0 else col_offset[ii - 1]

        offsets_scales = (
            row_offset,
            col_offset,
            scale_h,
            scale_w,
        )

        return offsets_scales, (h, w)

    def _sample_davis_video_frame(
        self, davis_vid_name: str, instance: int, frame_number: int
    ) -> tuple[NDArray]:
        # Initialize scale, starting point

        video_path = self.davis_annotations.joinpath(davis_vid_name)

        mask = video_path.joinpath("00000.png")
        mask = np.array(Image.open(mask).convert("P"))
        bbox_0 = convert_bbox_dims_to_width(get_bbox_start_end(mask))
        bbox_0 = np.array(bbox_0)

        frames = sorted(video_path.iterdir())

        frame = frames[frame_number]

        # get bbox of frame
        mask = np.array(Image.open(frame).convert("P"))
        mask = select_instance(mask, instance)
        bbox_i = convert_bbox_dims_to_width(get_bbox_start_end(mask))
        bbox_i = np.array(bbox_i)
        offsets = bbox_i[:2] - bbox_0[:2]
        scales = bbox_i[2:] / bbox_0[2:]

        offsets_scales = offsets[0], offsets[1], scales[0], scales[1]
        return offsets_scales

    def generate_fake_frames_masks(
        self, coco_frame: NDArray, coco_mask: NDArray, offsets_scales: tuple[NDArray]
    ) -> tuple[list[Image.Image]]:
        coco_frames = []
        coco_masks = []

        coco_frame = Image.fromarray(coco_frame)
        coco_mask = Image.fromarray(coco_mask)
        coco_frame = np.array(coco_frame)
        coco_mask = np.array(coco_mask)

        offset_row, offset_col, scale_h, scale_w = offsets_scales

        for i in range(len(offsets_scales[0])):

            if (
                offset_col[i] < (coco_mask.shape[1] // 2 + 10)
                or offset_col[i] >= coco_mask.shape[1] // 2 - 10
            ):
                offset_col[i] = -offset_col[i]

            if (
                offset_row[i] < (coco_mask.shape[0] // 2 + 10)
                or offset_row[i] >= coco_mask.shape[0] // 2 - 10
            ):
                offset_row[i] = -offset_row[i]

            affine_transf = (
                1,  # * scale_w[i],
                0,
                -offset_col[i] - 0 * coco_frame.shape[1],
                0,
                1,  # * scale_h[i],
                -offset_row[i] - 0 * coco_frame.shape[0],
            )
            new_mask = Image.fromarray(coco_mask)
            new_mask = new_mask.transform(
                new_mask.size, Image.AFFINE, data=affine_transf, fill=0
            )
            new_mask = np.array(new_mask)

            new_frame = Image.fromarray(coco_frame)
            new_frame = new_frame.transform(
                new_frame.size, Image.AFFINE, data=affine_transf, fill=0
            )
            new_frame = np.array(new_frame)
            new_frame[new_mask <= 0] = 0

            coco_masks.append(Image.fromarray(new_mask))
            coco_frames.append(Image.fromarray(new_frame))

        return coco_frames, coco_masks

    def get_augmentation_data(self, davis_video_name: str) -> tuple[list[Image.Image]]:

        davis_vid_name, instance = self._get_davis_video_name_instance(davis_video_name)
        offsets_scales, resize_shape_h_w = self._sample_davis_video(
            davis_vid_name, instance
        )
        coco_frame, coco_mask = self._get_coco_data(resize_shape_h_w)
        coco_frames, coco_masks = self.generate_fake_frames_masks(
            coco_frame, coco_mask, offsets_scales
        )

        return coco_frames, coco_masks
