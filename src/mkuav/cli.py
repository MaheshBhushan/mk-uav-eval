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
        cmd_parser = sub.add_parser(name, help=help_text)
        if name == "detect":
            cmd_parser.add_argument("image")
            cmd_parser.add_argument("--backend", choices=("cv2", "ort"), default="cv2")
            cmd_parser.add_argument("--model", default="models/yolov8n.onnx")
            cmd_parser.add_argument("--conf", type=float, default=0.25)
            cmd_parser.add_argument("--iou", type=float, default=0.5)
        if name == "bench":
            cmd_parser.add_argument("--model", default="models/yolov8n.onnx")
            cmd_parser.add_argument("--image", default="/var/tmp/bus.jpg")
            cmd_parser.add_argument("--iters", type=int, default=100)
            cmd_parser.add_argument("--warmup", type=int, default=20)
            cmd_parser.add_argument("--json", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0
    if args.cmd == "detect":
        from mkuav import detect

        return detect.main(args)
    if args.cmd == "bench":
        from mkuav import bench

        return bench.main(args)
    print(f"{args.cmd}: not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
