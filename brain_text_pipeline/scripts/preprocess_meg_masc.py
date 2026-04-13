#!/usr/bin/env python3
"""Preprocess MEG-MASC raw data into numpy arrays.

Outputs per-run files:
  out_dir/sub-XX/ses-YY/task-Z/brain.npy (T x D)
  out_dir/sub-XX/ses-YY/task-Z/meta.json

Options for band-pass, resampling, and z-scoring.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_text_pipeline.src.data.meg_masc import (
    list_sessions,
    list_subjects,
    list_tasks,
    load_raw_kit,
    meg_con_path,
    markers_path,
    headshape_paths,
)
from brain_text_pipeline.src.utils.logging import log

try:
    import mne
except ImportError as e:  # pragma: no cover
    raise SystemExit("mne is required for MEG preprocessing") from e


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="MEG-MASC dataset root")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--sfreq", type=float, default=200.0, help="Resample frequency")
    ap.add_argument("--l_freq", type=float, default=0.5)
    ap.add_argument("--h_freq", type=float, default=40.0)
    ap.add_argument("--zscore", action="store_true")
    args = ap.parse_args()

    subjects = args.subjects or list_subjects(args.root)
    if not subjects:
        raise SystemExit(f"No subjects found under {args.root}. Check --root path.")
    log(f"Found {len(subjects)} subjects under {args.root}")
    for sub in subjects:
        sessions = args.sessions or list_sessions(args.root, sub)
        if not sessions:
            log(f"No sessions for {sub}, skipping")
            continue
        for ses in sessions:
            tasks = args.tasks or list_tasks(args.root, sub, ses)
            if not tasks:
                log(f"No tasks for {sub}/{ses}, skipping")
                continue
            for task in tasks:
                con_path = meg_con_path(args.root, sub, ses, task)
                mrk_path = markers_path(args.root, sub, ses, task)
                hsp_path, elp_path = headshape_paths(args.root, sub, ses)
                if not con_path.exists():
                    log(f"Missing {con_path}, skipping")
                    continue
                log(f"Loading {con_path}")
                raw = load_raw_kit(con_path, mrk_path, elp_path, hsp_path)
                raw.pick_types(meg=True)
                raw.load_data()
                raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, fir_design="firwin", verbose=False)
                if args.sfreq:
                    raw.resample(args.sfreq)
                data = raw.get_data().T.astype(np.float32)
                if args.zscore:
                    mean = data.mean(axis=0, keepdims=True)
                    std = data.std(axis=0, keepdims=True) + 1e-6
                    data = (data - mean) / std
                else:
                    mean = None
                    std = None

                out_dir = args.out_dir / sub / ses / f"task-{task}"
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / "brain.npy", data)
                meta = {
                    "subject": sub,
                    "session": ses,
                    "task": task,
                    "sfreq": raw.info["sfreq"],
                    "n_channels": data.shape[1],
                    "n_times": data.shape[0],
                    "channels": raw.info["ch_names"],
                    "zscore": args.zscore,
                }
                if mean is not None:
                    np.save(out_dir / "mean.npy", mean.astype(np.float32))
                    np.save(out_dir / "std.npy", std.astype(np.float32))
                with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                log(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
