import torch
import torch.nn.functional as F

# from torch.utils.data import DataLoader
import numpy as np
from numpy.typing import NDArray

from PIL import Image

from model.eval_network import STCN

from davis_evaluation.davis2017.metrics import db_eval_boundary, db_eval_iou

# from dataset.davis_test_dataset import DAVISTestDataset
from util.tensor_util import unpad
from inference_core import InferenceCore

from progressbar import progressbar

from pathlib import Path


def get_gt_maks(
    gt_annotations_path: Path, video_name: str, frames: list[int]
) -> NDArray:
    video_path = gt_annotations_path.joinpath(video_name)

    tmp = video_path.joinpath(frames[0] + ".png")
    tmp = np.array(Image.open(tmp.as_posix()))

    masks = np.zeros((len(frames),) + tmp.shape)

    for ii, frame in enumerate(frames):

        frame_path = video_path.joinpath(frame + ".png")

        gt_mask = Image.open(frame_path).convert("P")

        masks[ii] = np.array(gt_mask)

    return masks.astype(np.uint8)


def prepare_mask_instance(mask: NDArray, instance: int) -> NDArray:
    mask_inst = np.zeros_like(mask).astype(np.uint8)  # Maybe bool?

    mask_inst[mask == instance] = 1

    return mask_inst


def evaluate_davis_sequence(all_gt_masks: NDArray, all_res_masks: NDArray) -> dict:

    all_void_masks = None

    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(
        all_gt_masks.shape[:2]
    )

    metric = ("J", "F")
    metrics_res = {}
    if "J" in metric:
        metrics_res["J"] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    if "F" in metric:
        metrics_res["F"] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    for ii in range(all_gt_masks.shape[0]):
        if "J" in metric:
            j_metrics_res[ii, :] = db_eval_iou(
                all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks
            )
        if "F" in metric:
            f_metrics_res[ii, :] = db_eval_boundary(
                all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks
            )
    return j_metrics_res, f_metrics_res


def online_davis_eval(
    test_loader,
    model,
    gt_annotations_path,
    top_k=20,
    mem_every: int = 5,
    include_last: bool = True,
    amp_enabled: bool = True,
    single_object: bool = True,
):

    torch.autograd.set_grad_enabled(False)

    prop_model = STCN().cuda().eval()

    # Performs input mapping such that stage 0 model can be loaded
    prop_saved = model.state_dict()
    for k in list(prop_saved.keys()):
        if k == "value_encoder.conv1.weight":
            if prop_saved[k].shape[1] == 4:
                pads = torch.zeros((64, 1, 7, 7), device=prop_saved[k].device)
                prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
    prop_model.load_state_dict(prop_saved)

    # Start eval
    ious_means = []
    fs_means = []
    for data in progressbar(
        test_loader, max_value=len(test_loader), redirect_stdout=True
    ):

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            rgb = data["rgb"].cuda()
            msk = data["gt"][0].cuda()
            info = data["info"]
            name = info["name"][0]
            k = len(info["labels"][0])
            size = info["size_480p"]

            processor = InferenceCore(
                prop_model,
                rgb,
                k,
                top_k=top_k,
                mem_every=mem_every,
                include_last=include_last,
            )
            processor.interact(msk[:, 0], 0, rgb.shape[1])

            # Do unpad -> upsample to original size
            out_masks = torch.zeros(
                (processor.t, 1, *size), dtype=torch.uint8, device="cuda"
            )
            for ti in range(processor.t):
                prob = unpad(processor.prob[:, ti], processor.pad)
                prob = F.interpolate(prob, size, mode="bilinear", align_corners=False)
                out_masks[ti] = torch.argmax(prob, dim=0)

            out_masks = (out_masks.detach().cpu().numpy()[:, 0]).astype(np.uint8)

            video_name = info["name"][0]
            instances = info["labels"].cpu().numpy()[0]
            frames = info["frames"]

            frames_ = [frame[0].split(".")[0] for frame in frames]

            # get gt_masks
            gt_masks = get_gt_maks(
                gt_annotations_path=Path(gt_annotations_path),
                video_name=video_name,
                frames=frames_,
            )

            if single_object:
                gt_masks_all = np.zeros((len(instances),) + gt_masks.shape)
                out_masks_all = np.zeros((len(instances),) + out_masks.shape)

                for ii, instance in enumerate(instances):
                    gt_masks_all[ii] = prepare_mask_instance(gt_masks, instance)
                    out_masks_all[ii] = prepare_mask_instance(out_masks, instance)

            else:
                ## Remember to unsqueeze(0)
                gt_masks_all = torch.from_numpy(gt_masks).unsqueeze(0).numpy()
                out_masks_all = torch.from_numpy(out_masks).unsqueeze(0).numpy()

            gt_masks_all = gt_masks_all[:, 1:-1, :, :]
            out_masks_all = out_masks_all[:, 1:-1, :, :]

            for inst in range(len(instances)):

                ious = np.zeros(gt_masks_all.shape[1])
                fs = np.zeros(gt_masks_all.shape[1])
                for frame in range(gt_masks_all.shape[1]):
                    ious[frame] = db_eval_iou(
                        gt_masks_all[inst, frame, :, :],
                        out_masks_all[inst, frame, :, :],
                    )

                    fs[frame] = db_eval_boundary(
                        gt_masks_all[inst, frame, :, :],
                        out_masks_all[inst, frame, :, :],
                    )

                ious_means.append(np.mean(ious))
                fs_means.append(np.mean(fs))

    return np.mean(np.array(ious_means)), np.mean(np.array(fs_means))
