import argparse

from ezflow.data import DataloaderCreator
from ezflow.engine import eval_model
from ezflow.models import build_model


def main():

    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to he data root dir"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to train"
    )
    parser.add_argument(
        "--model_cfg", type=str, required=False, help="Path to the model config file"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Model ckpt to be used",
    )
    parser.add_argument("--device", type=str, default="0", help="Device ID")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--crop_size", type=str, default="368,496", help="Crop size")
    parser.add_argument("--dataset", type=str, default="sintel", help="Dataset")

    args = parser.parse_args()

    val_loader_creator = DataloaderCreator(
        batch_size=args.bs,
        num_workers=1,
        pin_memory=True,
    )

    if args.dataset == "sintel":

        val_loader_creator.add_mpi_sintel(
            root_dir=args.data_dir,
            split="training",
            dstype="final",
            augment=True,
            aug_params={
                "crop_size": list(map(int, args.crop_size.split(","))),
                "spatial_aug_params": {"aug_prob": 0.0},
                "color_aug_params": {"aug_prob": 0.0},
                "eraser_aug_params": {"aug_prob": 0.0},
            },
        )

    else:

        val_loader_creator.add_flying_chairs(
            root_dir=args.data_dir,
            split="validation",
            augment=True,
            aug_params={
                "crop_size": list(map(int, args.crop_size.split(","))),
                "spatial_aug_params": {"aug_prob": 0.0},
                "color_aug_params": {"aug_prob": 0.0},
                "eraser_aug_params": {"aug_prob": 0.0},
            },
        )

    val_loader = val_loader_creator.get_dataloader()

    if args.model_cfg is not None:
        model = build_model(
            args.model, cfg_path=args.model_cfg, custom_cfg=True, weights_path=args.ckpt
        )
    else:
        model = build_model(args.model, default=True, weights_path=args.ckpt)

    eval_model(model, val_loader, args.device)


if __name__ == "__main__":

    main()
