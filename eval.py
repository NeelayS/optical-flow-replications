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
        "--model_cfg", type=str, required=True, help="Path to the model config file"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Model ckpt to be used",
    )
    parser.add_argument("--device", type=str, default="0", help="Device ID")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--crop", default=False, action="store_true", help="Whether to crop images"
    )
    parser.add_argument(
        "--crop_size", type=int, nargs="+", default=[256, 256], help="Crop size"
    )
    parser.add_argument("--dataset", type=str, default="sintel", help="Dataset")
    parser.add_argument(
        "--sintel_pass", type=str, default="clean", help="Sintel dataset pass"
    )
    parser.add_argument(
        "--sintel_split", type=str, default="training", help="Sintel dataset split"
    )
    parser.add_argument("--pad_divisor", type=int, default=8, help="Pad divisor")

    args = parser.parse_args()

    val_loader_creator = DataloaderCreator(
        batch_size=args.bs,
        num_workers=1,
        pin_memory=True,
    )

    if args.dataset.lower() == "sintel":

        print("Using MPI Sintel dataset for evaluation")

        val_loader_creator.add_MPISintel(
            root_dir=args.data_dir,
            split=args.sintel_split.lower(),
            dstype=args.sintel_pass.lower(),
            crop=args.crop,
            crop_type="center",
            crop_size=args.crop_size,
            augment=False,
        )

    else:

        print("Using Flying Chairs for evaluation")

        val_loader_creator.add_FlyingChairs(
            root_dir=args.data_dir,
            split="validation",
            crop=args.crop,
            crop_type="center",
            crop_size=args.crop_size,
            augment=False,
        )

    val_loader = val_loader_creator.get_dataloader()
    _, target = next(iter(val_loader))
    print(
        f"Processing images of spatial resolution: {target.shape[-2:]} with a padding (if required) of {args.pad_divisor}"
    )

    model = build_model(
        args.model, cfg_path=args.model_cfg, custom_cfg=True, weights_path=args.ckpt
    )

    eval_model(model, val_loader, args.device, pad_divisor=args.pad_divisor)


if __name__ == "__main__":

    main()
