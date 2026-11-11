#!/usr/bin/env python3
from src.cli import evaluation_parser, run_evaluation


def main():
    parser = evaluation_parser("Evaluate longitudinal FAF predictions.")
    run_evaluation(parser.parse_args())


if __name__ == "__main__":
    main()
