import argparse

from ezflow.models import build_model

from flow_transformer import FlowTv1


def main():

    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to train"
    )
    parser.add_argument(
        "--model_cfg", type=str, required=False, help="Path to the model config file"
    )

    args = parser.parse_args()

    if args.model_cfg is not None:
        model = build_model(
            args.model,
            cfg_path=args.model_cfg,
            custom_cfg=True,
        )
    else:
        model = build_model(args.model)

    print(
        f"No. of parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )


if __name__ == "__main__":

    main()
