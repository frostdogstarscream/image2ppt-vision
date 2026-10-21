"""Advisory threshold sweep over a labelled typography sample manifest.

Usage:
  python tools/calibrate_thresholds.py --samples samples.json --output report.json

The tool reads local images only. It never calls a model or changes defaults.
"""
import argparse
import json
from pathlib import Path

from image2ppt_vision.threshold_calibration import load_samples, sweep


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="TypographyThresholdSamples/1 JSON")
    parser.add_argument("--output", required=True, help="new report JSON path")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite an existing report: " + str(output))
    report = sweep(load_samples(args.samples))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "current": report["current"]["metrics"],
                      "recommended": None if report["recommended"] is None else
                      report["recommended"]["thresholds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
