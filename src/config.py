from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_root: Path = Path("/home/krschap/data")
    output_dir: Path = Path("outputs")
    seed: int = 64

    train_regions: list[str] = field(
        default_factory=lambda: [
            "ramp_dhaka_bangladesh",
            "ramp_barishal_bangladesh",
            "ramp_sylhet_bangladesh",
        ]
    )
    val_regions: list[str] = field(
        default_factory=lambda: ["ramp_coxs_bazar_bangladesh"]
    )
    test_regions: list[str] = field(default_factory=lambda: ["stage1_ramp_sample"])

    val_split: float = 0.2
    res: float = 0.4
    pretrained_model: str = (
        "facebook/mask2former-swin-base-IN21k-coco-instance"  # https://huggingface.co/facebook/mask2former-swin-base-IN21k-coco-instance
    )

    epochs: int = 10
    batch_size: int = 8
    num_samples: int = 5

    # hyper params
    dice_weight: float = 5.0
    mask_weight: float = 5.0
    class_weight: float = 5.0
    learning_rate: float = 0.00001
    weight_decay: float = 0.0001  # penalty on large weights to prevent overfitting
    early_stopping_patience: int = 5

    num_workers: int = 32
    use_wandb: bool = True
    wandb_project: str = "building-seg-mask2former"
    wandb_run_name: str = "default_run"
