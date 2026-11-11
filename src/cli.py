import argparse
from pathlib import Path

import torch

from .clinical_metrics import load_segmentor
from .data import sequences, write_json
from .evaluation import evaluate, prediction_sources, run_metadata
from .predictors import add_model_arguments, load_models
from .reporting import write_results


def evaluation_parser(description):
    parser = argparse.ArgumentParser(description=description)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--manifest")
    data.add_argument("--data_dir", "--faf_holdout_dir", dest="data_dirs", action="append")
    parser.add_argument("--split")
    parser.add_argument("--cohort", "--faf_holdout_name")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--segmentor_weights")
    parser.add_argument("--seg_module")
    parser.add_argument("--fovea_lookup")
    parser.add_argument("--seg_border_px", type=int, default=0)
    parser.add_argument("--predictions_csv")
    parser.add_argument("--save_probabilities", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    add_model_arguments(parser)
    return parser


def run_evaluation(args, k_samples=1):
    if args.manifest and not args.split:
        raise ValueError("Select a manifest split for evaluation.")
    if args.gpu is not None:
        args.device = f"cuda:{args.gpu}"
    seqs = sequences(args.data_dirs, args.manifest, args.split, args.cohort)
    cohort = args.cohort or next(iter(seqs.values()))[0]["cohort"]
    models = load_models(args, torch.device(args.device))
    segmentor = load_segmentor(args.segmentor_weights, args.seg_module, args.device)
    frame = evaluate(seqs, models, args.output_dir, segmentor, args.fovea_lookup,
                     prediction_sources(args.predictions_csv), cohort, args.seed,
                     args.save_probabilities, args.save_predictions, args.seg_border_px, k_samples)
    output = Path(args.output_dir)
    write_results(frame, output, args.bootstrap, args.seed)
    metadata = run_metadata(args, seqs)
    metadata["k_samples"] = k_samples
    metadata["history_limits"] = {
        name: dict(inference=predict.max_history, training=predict.training_max_history)
        for name, predict in models.items() if hasattr(predict, "training_max_history")}
    write_json(output / "run.json", metadata)
    return frame
