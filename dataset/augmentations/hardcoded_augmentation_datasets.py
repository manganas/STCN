from pathlib import Path


def get_augmentation_datasets_paths(augmentation_datasets: list[str]) -> dict[str:Path]:
    """
    Hardcoded paths. At least keep them in one place and remove responsibility from
    augmentations class.

    I could read the paths from a json or text document instead.
    """
    available_datasets = {
        "davis": "/work3/s220493/DAVIS/2017/trainval",
        "fss": "/work3/s220493/static/fss/",
        "ecssd": "/work3/s220493/static/ecssd/",
        "HRSOD": "/work3/s220493/static/HRSOD_small/",
        "DUTS-TR": "/work3/s220493/static/DUTS-TR/",
        "DUTS-TE": "/work3/s220493/static/DUTS-TE/",
        "BIG_small": "/work3/s220493/static/BIG_small/",
        "coco": "/work3/s220493/coco/",
    }

    results = {}
    for dataset in augmentation_datasets:
        if dataset in available_datasets.keys():
            results[dataset] = Path(available_datasets[dataset])
        else:
            # check if the path exists and has content. Now what content it is, we are adults
            if Path(dataset).is_dir() and any(Path(dataset).iterdir()):
                results[dataset] = Path(dataset)
                print(f"Using data from {dataset}")
            else:
                print(f"{dataset} not found or is empty.")

    print(f"{len(results)} accepted datasets:")
    for k, v in results.items():
        print(k, ": ", v)
    print()
    return results
