"""Command-line entry point. Subcommands are filled in by later modules."""
import argparse
import sys

SUBCOMMANDS = {
    "qa": "validate annotations for a dataset version",
    "detect": "run detection on an image",
    "segment": "run segmentation on an image",
    "eval-det": "score detection on a dataset version",
    "eval-seg": "score segmentation on a dataset version",
    "bench": "measure inference latency per backend",
    "robust": "mAP under perturbations",
    "report": "write report.md and metrics.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mkuav", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")
    for name, help_text in SUBCOMMANDS.items():
        sub.add_parser(name, help=help_text)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0
    print(f"{args.cmd}: not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
