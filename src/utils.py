import random
import warnings
from pathlib import Path
from typing import Any

import kornia.augmentation as K
import numpy as np
import torch
from kornia.constants import Resample
from torchgeo.datasets import RasterDataset, VectorDataset
from transformers import Mask2FormerImageProcessor

warnings.filterwarnings("ignore", category=UserWarning, module="kornia") # ignore the warnings of aligh_corners 



def get_augmentation():
    
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5, same_on_batch=False),
        K.RandomVerticalFlip(p=0.5, same_on_batch=False),
        K.RandomRotation(degrees=90, resample=Resample.BILINEAR, align_corners=False, same_on_batch=False, p=0.5),
        K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, same_on_batch=False, p=0.5),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), border_type='reflect', same_on_batch=False, p=0.3),
        data_keys=None,
        keepdim=True,
    )


class RAMPImageDataset(RasterDataset):
    filename_glob = "*.tif"
    is_image = True
    all_bands = ("R", "G", "B")
    rgb_bands = ("R", "G", "B")


class RAMPMaskDataset(VectorDataset):
    filename_glob = "*.geojson"

    def __init__(self, paths, crs=None, **kwargs):
        super().__init__(
            paths=paths, crs=crs, task="instance_segmentation", **kwargs
        ) # needs fixes from https://github.com/kshitijrajsharma/torchgeo/blob/8ff64198f6bbf0355986b43981fc5a2bbc008846/torchgeo/datasets/geo.py


def get_ramp_dataset(root: Path, regions: list[str]):

    image_paths, label_paths = [], []
    print(f"Finding image,label path for {regions}...")
    for region in regions:
        region_path = root / region
        img_path, lbl_path = region_path / "source", region_path / "labels"
        if img_path.exists() and lbl_path.exists():
            image_paths.append(img_path)
            label_paths.append(lbl_path)

    if not image_paths:
        raise ValueError(f"No valid regions found in {root}")

    print("Loading images ...")
    images = RAMPImageDataset(paths=image_paths)
    print(
        f"Loaded {len(images)} image tiles. using crs : {images.crs} with res {images.res}"
    )
    print("Loading labels ...")
    masks = RAMPMaskDataset(paths=label_paths)

    print(
        f"Loaded {len(masks)} mask tiles. using crs : {masks.crs} with res {masks.res}"
    )
    return images & masks


def get_all_ramp_regions(root: Path) -> list[str]:
    regions = [
        d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("ramp_")
    ]
    if not regions:
        raise ValueError(f"No RAMP regions found in {root}")
    return sorted(regions)


def split_regions(regions: list[str], val_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    shuffled = regions.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def get_image_processor(
    pretrained_model: str, size: int = 256
) -> Mask2FormerImageProcessor:
    return Mask2FormerImageProcessor.from_pretrained(
        pretrained_model,
        num_labels=2,
        # do_reduce_labels=True,
        ignore_index=255,
        size=size,
        do_normalize=True,
    )


def make_collate_fn(image_processor: Mask2FormerImageProcessor):
    def collate_fn(
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:  # source : https://debuggercafe.com/fine-tuning-mask2former/
        images = [sample["image"].permute(1, 2, 0).numpy().astype('uint8') for sample in batch]
        inputs = image_processor(images=images, return_tensors="pt")

        mask_labels = []
        class_labels = []

        for sample in batch:
            mask = sample["mask"]
            
            # edge case : dataset returns 2D mask when no instances exist
            if mask.ndim == 2:
                H, W = mask.shape
                mask_labels.append(torch.zeros((0, H, W), dtype=torch.float32))
                class_labels.append(torch.tensor([], dtype=torch.long))
                continue

            instance_masks = []
            instance_classes = []

            for i in range(mask.shape[0]):
                instance_mask = mask[i]
                if instance_mask.sum() > 0:
                    instance_masks.append(instance_mask.float())
                    instance_classes.append(1)

            if instance_masks:
                mask_labels.append(torch.stack(instance_masks))
                class_labels.append(torch.tensor(instance_classes, dtype=torch.long))
            else:
                H, W = mask.shape[-2:]
                mask_labels.append(torch.zeros((0, H, W), dtype=torch.float32))
                class_labels.append(torch.tensor([], dtype=torch.long))

        inputs["mask_labels"] = mask_labels
        inputs["class_labels"] = class_labels
        return inputs

    return collate_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)