#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import evaluation_parser, run_evaluation


def main():
    parser = evaluation_parser("Sample aligned predictors under fixed longitudinal histories.")
    parser.add_argument("--k_samples", type=int, default=10)
    args = parser.parse_args()
    if args.k_samples < 2:
        parser.error("At least two samples are required.")
    if not (args.ckpt_ia_nonlinear or args.ckpt_ia_linear):
        parser.error("Supply an aligned stochastic checkpoint.")
    if not args.ckpt_tru:
        parser.error("Supply the TRU checkpoint for paired endpoint-gain comparisons.")
    run_evaluation(args, k_samples=args.k_samples)


if __name__ == "__main__":
    main()
