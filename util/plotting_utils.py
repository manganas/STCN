from pathlib import Path
import matplotlib.pyplot as plt


def plot_frame_of_batch(
    data: dict, frame_number=0, batch_element=0, name: str = "frame_content.png"
) -> None:
    """
    data = {
            "rgb": images,
            "gt": tar_masks,
            "cls_gt": cls_gt,
            "sec_gt": sec_masks,
            "selector": selector,
            "info": info,
        }

    data is the yield output of the dataloader object
    """
    _path_ = Path("./plots")
    _path_.mkdir(exist_ok=True)

    frames = data["rgb"]
    gt = data["gt"]
    cls_gt = data["cls_gt"]
    sec_gt = data["sec_gt"]

    # Take the first of the batch
    frames = data["rgb"][batch_element]
    gt = data["gt"][batch_element]
    cls_gt = data["cls_gt"][batch_element]
    sec_gt = data["sec_gt"][batch_element]

    # Take the first frame
    frames = frames[frame_number]
    gt = gt[frame_number]
    cls_gt = cls_gt[frame_number]
    sec_gt = sec_gt[frame_number]

    frames = frames.permute(1, 2, 0)
    gt = gt.permute(1, 2, 0)
    sec_gt = sec_gt.permute(1, 2, 0)

    fig, ax = plt.subplots(1, 4)
    fig.suptitle(f"Batch element: {batch_element}\n Frame number: {frame_number}")

    ax[0].imshow(frames)
    ax[0].set_title("frames")
    ax[0].axis("off")

    ax[1].imshow(gt)
    ax[1].set_title("gt")
    ax[1].axis("off")

    ax[2].imshow(cls_gt)
    ax[2].set_title("cls_gt")
    ax[2].axis("off")

    ax[3].imshow(sec_gt)
    ax[3].set_title("sec_gt")
    ax[3].axis("off")

    fig.tight_layout()

    path = Path.joinpath(_path_, name)
    plt.savefig(path)
    plt.show()
