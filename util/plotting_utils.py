def plot_first_frame_of_batch(data: dict, path: str = "frame_content.png") -> None:
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
    frames = data["rgb"]
    gt = data["gt"]
    cls_gt = data["cls_gt"]
    sec_gt = data["sec_gt"]

    # Take the first of the batch
    frames = data["rgb"][0]
    gt = data["gt"][0]
    cls_gt = data["cls_gt"][0]
    sec_gt = data["sec_gt"][0]

    # Take the first frame
    frames = data["rgb"][0][0]
    gt = data["gt"][0][0]
    cls_gt = data["cls_gt"][0][0]
    sec_gt = data["sec_gt"][0][0]

    print(frames.shape, gt.shape, cls_gt.shape, sec_gt.shape)
    import matplotlib.pyplot as plt

    frames = frames.permute(1, 2, 0)
    gt = gt.permute(1, 2, 0)
    sec_gt = sec_gt.permute(1, 2, 0)

    fig, ax = plt.subplots(1, 4)
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

    plt.savefig(path)
    plt.show()
