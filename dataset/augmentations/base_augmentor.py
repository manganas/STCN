from pathlib import Path
import numpy as np

from dataset.augmentations.augmentors import (
    DAVISAugmentor,
    YTAugmentor,
    COCOAugmentor,
    StaticAugmentor,
    FSSAugmentor,
)

from PIL import Image


class Augmentor:
    def __init__(self, datasets: list[str], probabilities: list[float] = None):

        # Initialize the augmentors here, since for COCO it takes around 14 seconds
        # for the index to be built.

        data_paths_base = Path("/work3/s220493/")
        static_paths_base = data_paths_base.joinpath("static")
        coco_path = data_paths_base.joinpath("coco")
        davis_path = data_paths_base.joinpath(
            "DAVIS", "2017", "trainval"
        )  # for vos type datasets, also ytvos

        # Hardcoded paths for now!
        SUPPORTED_DATASETS = {
            "BIG_small": static_paths_base.joinpath("BIG_small"),  # hyma
            "DUTS-TE": static_paths_base.joinpath("DUTS-TE"),  # hyma
            "DUTS-TR": static_paths_base.joinpath("DUTS-TR"),  # hyma
            "HRSOD_small": static_paths_base.joinpath("HRSOD_small"),  # hyma
            "ecssd": static_paths_base.joinpath("ecssd"),  # hyma
            "fss": static_paths_base.joinpath("fss"),  # in folders
            "coco": coco_path,
            "davis": davis_path,
            "yt": data_paths_base.joinpath("YouTube", "train_480p"),
        }

        ##

        self.datasets = datasets  # list of strings for each dataset

        for dataset in datasets:
            if dataset not in list(SUPPORTED_DATASETS.keys()):
                raise NotImplementedError(
                    "Dataset name passed in Augmentor is not supported!"
                )

        if probabilities:
            assert len(datasets) == len(
                probabilities
            ), "Not the same number of probabilities and datasets for augmentation passed!"
            self.probabilities = probabilities
        else:
            self.probabilities = [1 // len(datasets)] * len(self.datasets)

        # davis, coco, static, fss (not hyma)
        self.augmentors = {}

        for dataset in datasets:

            if dataset in ["coco"]:
                tmp_augm = self.augmentors.get("coco", None)
                if not tmp_augm:
                    self.augmentors["coco"] = COCOAugmentor(coco_path, davis_path)
            elif dataset in ["davis"]:
                tmp_augm = self.augmentors.get("davis", None)
                if not tmp_augm:
                    self.augmentors["davis"] = DAVISAugmentor(davis_path)
            elif dataset in ["yt"]:
                tmp_augm = self.augmentors.get("yt", None)
                if not tmp_augm:
                    self.augmentors["yt"] = YTAugmentor(SUPPORTED_DATASETS["yt"])
            elif dataset in ["fss"]:
                tmp_augm = self.augmentors.get("fss", None)
                if not tmp_augm:
                    self.augmentors["fss"] = FSSAugmentor(
                        SUPPORTED_DATASETS["fss"], davis_path
                    )
            else:
                tmp_augm = self.augmentors.get(dataset, None)
                if not tmp_augm:
                    self.augmentors[dataset] = StaticAugmentor(
                        SUPPORTED_DATASETS[dataset], davis_path
                    )

        ##### Varying members per __getitem__ call
        self.selected_datasets = []
        self.selected_augmentors = []

    def select_datasets(self, num_of_augmentations: int, replace: bool = True) -> None:
        self.selected_datasets = np.random.choice(
            self.datasets, size=num_of_augmentations, replace=replace
        )

        return self.selected_datasets

    def set_augmentors(self) -> None:
        self.selected_augmentors = []
        for dataset in self.selected_datasets:
            self.selected_augmentors.append(self.augmentors[dataset])

        return

    def get_augmentation_data(
        self, augmentation_idx: int
    ) -> tuple[list[Path], list[Path]]:
        augmentor = self.selected_augmentors[augmentation_idx]

        # Get a random data point
        images_paths, masks_path = augmentor.get_augmentation_data()

        return images_paths, masks_path
