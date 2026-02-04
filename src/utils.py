import random
import warnings
from dataclasses import dataclass
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
        f"Loaded {len(images)} image tiles."
    )
    print("Loading labels ...")
    masks = RAMPMaskDataset(paths=label_paths)

    print(
        f"Loaded {len(masks)} mask tiles."
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


@dataclass
class DatasetStats:
    num_samples: int
    total_buildings: int
    avg_buildings_per_image: float
    buildings_range: tuple[int, int]
    avg_building_area_px: float
    empty_mask_ratio: float
    

def validate_dataset(dataset, verbose=False):
    building_counts = []
    areas = []
    
    for idx, sample in enumerate(dataset):
        mask = sample['mask']
        
        if mask.ndim == 2 or mask.sum() == 0:
            building_counts.append(0)
            if verbose and idx < 5:
                print(f"  Sample {idx}: Empty mask")
            continue
        
        num_buildings = mask.shape[0]
        building_counts.append(num_buildings)
        areas.extend([mask[i].sum().item() for i in range(num_buildings) if mask[i].sum() > 0])
        
        if verbose and idx < 5:
            print(f"  Sample {idx}: {num_buildings} buildings")
    
    building_counts_np = np.array(building_counts)
    areas_np = np.array(areas) if areas else np.array([0])
    
    stats = DatasetStats(
        num_samples=len(dataset),
        total_buildings=int(building_counts_np.sum()),
        avg_buildings_per_image=float(building_counts_np.mean()),
        buildings_range=(int(building_counts_np.min()), int(building_counts_np.max())),
        avg_building_area_px=float(areas_np.mean()),
        empty_mask_ratio=float((building_counts_np == 0).sum() / len(dataset)),
    )
    
    if verbose:
        print(f"\nDataset: {stats.num_samples} samples, {stats.total_buildings} buildings")
        print(f"Avg buildings/img: {stats.avg_buildings_per_image:.2f}, range: {stats.buildings_range}")
        print(f"Avg area: {stats.avg_building_area_px:.1f}px, empty: {stats.empty_mask_ratio:.1%}\n")
    
    return stats


def visualize_attention_maps(model, batch, query_idx=0, save_path=None):
    import matplotlib.pyplot as plt
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        outputs = model(batch["pixel_values"].to(device))
        
        if not (hasattr(outputs, 'attentions') and outputs.attentions is not None):
            print(f"No attention maps available. Output keys: {getattr(outputs, 'keys', lambda: 'N/A')()}")
            return None
        
        img = batch["pixel_values"][0].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        attn = outputs.attentions[-1][0, query_idx].cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(img)
        ax1.set_title("Input Image")
        ax1.axis('off')
        
        ax2.imshow(attn, cmap='hot', interpolation='nearest')
        ax2.set_title(f"Query {query_idx} Attention")
        ax2.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig