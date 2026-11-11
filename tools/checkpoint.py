#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import checksum, write_json
from src.backbone import MultiScaleTemporalUNet_v13
import torch


CONFIG_KEYS = ("method", "max_history", "image_size", "epochs", "batch_size", "lr",
               "warmup_steps", "history_dropout", "t_fixed", "t_infer", "t_min", "t_max",
               "ia_linear_t_min", "ia_linear_t_max", "ia_linear_t_infer", "seed", "val_split")


def main():
    parser = argparse.ArgumentParser(description="Inspect or export a TRU-family inference checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", choices=["tru", "std_ddim", "ia_nonlinear", "ia_linear"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--export_weights")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("args", {})
    config = config if isinstance(config, dict) else vars(config)
    clean = {k: config[k] for k in CONFIG_KEYS if k in config}
    clean["method"] = args.method
    model = MultiScaleTemporalUNet_v13(max_history=clean.get("max_history", 6))
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    report = dict(method=args.method, source_sha256=checksum(args.checkpoint),
                  epoch=checkpoint.get("epoch"), config=clean,
                  completed_epochs=int(checkpoint["epoch"]) + 1 if "epoch" in checkpoint else None,
                  parameters=sum(p.numel() for p in model.parameters()),
                  tensor_keys=len(checkpoint["ema_model"]))
    if args.export_weights:
        destination = Path(args.export_weights)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(ema_model=checkpoint["ema_model"], args=clean,
                        epoch=checkpoint.get("epoch")), destination)
        report.update(export_filename=destination.name, export_sha256=checksum(destination))
    write_json(args.output, report)


if __name__ == "__main__":
    main()
