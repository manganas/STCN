from pathlib import Path
import numpy as np
from PIL import Image

from dataset.augmentations.augmentors import (
    DAVISAugmentor,
    YTAugmentor,
    COCOAugmentor,
    StaticAugmentor,
)


class AugmentationDataGenerator:

    def get_n_successive_augm(prob_lists: list[float]) -> int:

        out_list = []

        for i in range(len(prob_lists)):
            a = np.random.rand()
            if a < prob_lists[i]:
                if i == 0 or i - 1 in out_list:
                    out_list.append(i)
                else:
                    break

        return len(out_list)

    def calculate_nested_probabilities(augm_probs: list[float]) -> list[float]:
        cumprob = []
        result = []
        augm_probs = sorted(augm_probs, reverse=True)
        for i in range(len(augm_probs)):
            if i == 0:
                cumprob.append(augm_probs[i])
                result.append(augm_probs[i])
                continue

            res = augm_probs[i] / cumprob[i - 1]

            result.append(res)
            cumprob.append(cumprob[i - 1] * res)

        return result

    # def __init__(self, datasets: dict[str:Path], probabilities: list[float] = None):
    def __init__(self, datasets: dict[str:Path], davis_root: Path):

        self.datasets = datasets

        # if probabilities:
        #     assert len(datasets) == len(
        #         probabilities
        #     ), "Not the same number of probabilities and datasets for augmentation passed!"
        #     self.probabilities = probabilities
        # else:
        #     self.probabilities = [1 // len(datasets)] * len(datasets)

        self.augmentors = {}

        for dataset in datasets.keys():

            if dataset == "davis":
                self.augmentors["davis"] = DAVISAugmentor(datasets["davis"])
            elif dataset == "yt":
                self.augmentors["yt"] = YTAugmentor(datasets["yt"])
            elif dataset == "coco":
                self.augmentors["coco"] = COCOAugmentor(datasets[dataset], davis_root)
            else:

                self.augmentors[dataset] = StaticAugmentor(
                    datasets[dataset], davis_root
                )

        ##### Varying members per __getitem__ call
        self._selected_datasets = []
        self._selected_augmentors = []

    def _select_datasets(self, num_of_augmentations: int, replace: bool = True) -> None:

        if not replace and num_of_augmentations > len(self.datasets):
            print(
                "Number of augmentations larger than number of available datasets without replace. Choice with replace."
            )
            replace = True

        self._selected_datasets = np.random.choice(
            list(self.datasets.keys()), size=num_of_augmentations, replace=replace
        )
        return

    def select_augmentors(
        self, num_of_augmentations: int, replace: bool = True
    ) -> None:
        self._selected_datasets = []
        self._select_datasets(num_of_augmentations, replace)

        self._selected_augmentors = []
        for dataset in self._selected_datasets:
            self._selected_augmentors.append(self.augmentors[dataset])
        return

    def get_augmentation_data(
        self, augmentation_idx: int
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        augmentor = self._selected_augmentors[augmentation_idx]

        # Get a random data point
        augm_frames, augm_masks = augmentor.get_augmentation_data()

        return augm_frames, augm_masks
