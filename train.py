import argparse

from ezflow.data import DataloaderCreator
from ezflow.engine import Trainer, get_training_cfg
from ezflow.models import build_model


def main():

    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument(
        "--train_cfg", type=str, required=True, help="Path to the training config file"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to train"
    )
    parser.add_argument(
        "--model_cfg", type=str, required=False, help="Path to the model config file"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory where logs are to be written",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory where ckpts are to be saved",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="Whether to resume training from a previous ckpt",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to ckpt for resuming training",
    )
    parser.add_argument(
        "--resume_epochs",
        type=int,
        default=None,
        help="Number of epochs to train after resumption",
    )
    parser.add_argument("--device", type=str, default="0", help="Device ID")
    parser.add_argument(
        "--distributed",
        type=bool,
        default=False,
        help="Whether to do distributed training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="FlyingChairs",
        help="Dataset to use",
    )
    parser.add_argument("--lr", type=float, required=False, help="Learning rate")

    args = parser.parse_args()

    training_cfg = get_training_cfg(cfg_path=args.train_cfg)
    training_cfg.LOG_DIR = args.log_dir
    training_cfg.CKPT_DIR = args.ckpt_dir
    training_cfg.DEVICE = args.device
    training_cfg.DISTRIBUTED = args.distributed

    if args.lr is not None:
        training_cfg.OPTIMIZER.LR = args.lr

    train_loader_creator = DataloaderCreator(
        batch_size=training_cfg.DATA.BATCH_SIZE,
        num_workers=training_cfg.DATA.NUM_WORKERS,
        pin_memory=training_cfg.DATA.PIN_MEMORY,
    )
    val_loader_creator = DataloaderCreator(
        batch_size=training_cfg.DATA.BATCH_SIZE,
        num_workers=training_cfg.DATA.NUM_WORKERS,
        pin_memory=training_cfg.DATA.PIN_MEMORY,
    )

    if args.dataset == "FlyingChairs":

        train_loader_creator.add_flying_chairs(
            root_dir="../../../Datasets/FlyingChairs_release/data",
            augment=training_cfg.DATA.AUGMENTATION.USE,
            aug_params={
                "crop_size": training_cfg.DATA.AUGMENTATION.PARAMS.CROP_SIZE,
                "spatial_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.TRAINING.SPATIAL_AUG_PARAMS,
            },
        )
        val_loader_creator.add_flying_chairs(
            root_dir="../../../Datasets/FlyingChairs_release/data",
            split="validation",
            augment=training_cfg.DATA.AUGMENTATION.USE,
            aug_params={
                "crop_size": training_cfg.DATA.AUGMENTATION.PARAMS.CROP_SIZE,
                "spatial_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.VALIDATION.SPATIAL_AUG_PARAMS,
                "color_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.VALIDATION.COLOR_AUG_PARAMS,
                "eraser_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.VALIDATION.ERASER_AUG_PARAMS,
            },
        )

    elif args.dataset == "FlyingThings3D":

        train_loader_creator.add_flying_things3d(
            root_dir="../../../Datasets/SceneFlow/FlyingThings3D",
            augment=training_cfg.DATA.AUGMENTATION.USE,
            aug_params={
                "crop_size": training_cfg.DATA.AUGMENTATION.PARAMS.CROP_SIZE,
                "spatial_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.TRAINING.SPATIAL_AUG_PARAMS,
            },
        )
        val_loader_creator.add_flying_things3d(
            root_dir="../../../Datasets/SceneFlow/FlyingThings3D",
            split="validation",
            augment=training_cfg.DATA.AUGMENTATION.USE,
            aug_params={
                "crop_size": training_cfg.DATA.AUGMENTATION.PARAMS.CROP_SIZE,
                "spatial_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.VALIDATION.SPATIAL_AUG_PARAMS,
                "color_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.VALIDATION.COLOR_AUG_PARAMS,
                "eraser_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.VALIDATION.ERASER_AUG_PARAMS,
            },
        )

    train_loader = train_loader_creator.get_dataloader()
    val_loader = val_loader_creator.get_dataloader()

    if args.model_cfg is not None:
        model = build_model(args.model, cfg_path=args.model_cfg, custom_cfg=True)
    else:
        model = build_model(args.model, default=True)

    trainer = Trainer(training_cfg, model, train_loader, val_loader)

    if args.resume:
        assert (
            args.resume_ckpt is not None
        ), "Please provide a ckpt to resume training from"
        trainer.resume_training(args.resume_ckpt, n_epochs=args.resume_epochs)

    else:
        trainer.train(n_epochs=args.epochs)


if __name__ == "__main__":

    main()
