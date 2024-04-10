from pathlib import Path

import numpy as np
from PIL import Image


class StaticImagesVOSAugmentations:
    def __init__(self, dataset_paths: list[Path] | list[str], sampling_datasets_paths=list[Path]|list[str]):
        
        self.sampling_datasets_paths = [Path(pth) for pth in sampling_datasets_paths]
        
        dataset_paths = [Path(pth) for pth in dataset_paths]

        # supported datasets
        supported_datasets = [
            "coco",
            "BIG_small",
            "DUTS-TE",
            "DUTS-TR",
            "HRSOD_small",
            "ecssd",
            "fss",
        ]
        
        self.used_datasets = []
        for dataset in dataset_paths:
            if dataset.stem not in supported_datasets:
                print(f"** Dataset \'{dataset.stem}\' at path {dataset} is not supported")
                continue
            used_datasets.append(dataset)

        print(f"Used datasets: ")
        for dataset in used_datasets:
            print(dataset.stem)
        
        # treat each dataset as it has been saved on disk
        self.augmentors:{}
        for dataset in used_datasets:
            if dataset.stem=='coco':
                self.augmentors['coco'] = COCOAugmentor(dataset)
            elif dataset.stem=='fss':
                self.augmentors['fss'] = FSSAugmentor(dataset)
            else:
                self.augmentors['standard'] = StandardStaticAugmentor(dataset)


    def select_motion_sampling_dataset(self):
        if len(self.sampling_datasets_paths)==0:
            print("There are no accepted motion sampling datasets!")
            raise NotImplementedError
        
        return np.random.choice(self.sampling_datasets_paths)[0]
    
    def select_static_dataset(self):
        if len(self.used_datasets)==0:
            print("There are no accepted augmentationdatasets!")
            raise NotImplementedError
        
        return np.random.choice(self.used_datasets,)[0]
    
    
            


def main():
    coca_dataset_path = '/work3/s220493/coca'
    coco_dataset_path = '/work3/s220493/coco'
    fss_dataset_path = '/work3/s220493/static/fss'
    ecssd_dataset_path = '/work3/s220493/ecssd'

    datasets = [coca_dataset_path, coco_dataset_path, fss_dataset_path, ecssd_dataset_path]

    static_augm = StaticImagesVOSAugmentations(datasets)


if __name__=='__main__':
    main()