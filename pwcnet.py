from ezflow.data import DataloaderCreator
from ezflow.engine import Trainer, get_training_cfg
from ezflow.models import build_model


train_loader = DataloaderCreator(batch_size=16, num_workers=1, pin_memory=True)
train_loader.add_flying_chairs(
    root_dir="../../../Datasets/FlyingChairs_release/data",
    augment=True,
    aug_params={
        "crop_size": (256, 256),
        "spatial_aug_params": {"min_scale": -0.1, "max_scale": 1.0, "flip": True},
    },
)
train_loader = train_loader.get_dataloader()

val_loader = DataloaderCreator(batch_size=16, num_workers=1, pin_memory=True)
val_loader.add_flying_chairs(
    root_dir="../../../Datasets/FlyingChairs_release/data",
    split="validation",
    augment=True,
    aug_params={
        "crop_size": (256, 256),
        "spatial_aug_params": {"aug_prob": 0.0},
        "color_aug_params": {"aug_prob": 0.0},
        "eraser_aug_params": {"aug_prob": 0.0},
    },
)
val_loader = val_loader.get_dataloader()

model = build_model("PWCNet", default=True)

training_cfg = get_training_cfg(cfg_path="configs/pwcnet_trainer.yaml")
training_cfg.DEVICE = 0
training_cfg.DISTRIBUTED = False
training_cfg.VAL_INTERVAL = 1
training_cfg.LOG_DIR = "./logs/pwcnet/run1"
training_cfg.CKPT_DIR = "./ckpts/pwcnet/run1"

trainer = Trainer(training_cfg, model, train_loader, val_loader)
trainer.train(n_epochs=25)

# trainer = Trainer(training_cfg, model, train_loader, val_loader)
# trainer.resume_training("./ckpts/run6/pwcnet_epochs5.pth", n_epochs=5)
