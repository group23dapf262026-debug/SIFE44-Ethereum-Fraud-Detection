"""Replay the Direct and Shuffle results from the saved TEST fit metrics."""

import argparse
from pathlib import Path
import sys

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/experiments/followup/tools"))
from paired_analysis import analyze


SORT_KEYS = {
    "summary": ["dataset", "model", "arm", "metric"],
    "split_means": ["dataset", "model", "arm", "split_seed"],
    "paired_deltas": ["dataset", "model", "candidate", "control", "split_seed"],
    "corrected_sensitivity": ["dataset", "model", "candidate", "control"],
}


def replay(output):
    ratios = {"Phish": 553 / 1656, "Ponzi": 269 / 804}
    for name, expected_fits, expected_contrasts in (
        ("direct", 5000, 40),
        ("shuffle", 3000, 30),
    ):
        source = ROOT / "results/followup" / name
        destination = output / name
        if destination.resolve() == source.resolve():
            raise ValueError("Choose an output directory outside the saved results.")
        fits = pd.read_csv(source / "test_metrics.csv", float_precision="round_trip")
        if len(fits) != expected_fits or set(fits.role) != {"test"}:
            raise ValueError(f"Unexpected TEST fit records for {name}")
        contrasts = analyze(fits, name, ratios, destination)
        if len(contrasts) != expected_contrasts:
            raise ValueError(f"Unexpected comparison count for {name}")
        for table, keys in SORT_KEYS.items():
            def load(directory):
                data = pd.read_csv(directory / f"{table}.csv", float_precision="round_trip")
                return data.sort_values(keys).reset_index(drop=True)
            assert_frame_equal(load(destination), load(source), rtol=1e-10, atol=1e-10)
        print(f"{name.capitalize()}: {expected_fits:,} TEST fits and "
              f"{expected_contrasts} paired comparisons verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "generated/followup")
    replay(parser.parse_args().output)
