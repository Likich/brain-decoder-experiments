"""Utilities for MEG-MASC dataset loading."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import mne
except ImportError:  # pragma: no cover
    mne = None


def list_subjects(root: Path) -> List[str]:
    return sorted([p.name for p in root.glob("sub-*") if p.is_dir()])


def list_sessions(root: Path, subject: str) -> List[str]:
    return sorted([p.name for p in (root / subject).glob("ses-*") if p.is_dir()])


def list_tasks(root: Path, subject: str, session: str) -> List[str]:
    meg_dir = root / subject / session / "meg"
    if not meg_dir.exists():
        return []
    tasks = []
    for f in meg_dir.glob(f"{subject}_{session}_task-*_events.tsv"):
        # filename: sub-XX_ses-YY_task-<id>_events.tsv
        parts = f.name.split("_")
        for part in parts:
            if part.startswith("task-"):
                tasks.append(part.replace("task-", ""))
    return sorted(set(tasks))


def events_path(root: Path, subject: str, session: str, task: str) -> Path:
    return root / subject / session / "meg" / f"{subject}_{session}_task-{task}_events.tsv"


def meg_con_path(root: Path, subject: str, session: str, task: str) -> Path:
    return root / subject / session / "meg" / f"{subject}_{session}_task-{task}_meg.con"


def markers_path(root: Path, subject: str, session: str, task: str) -> Path:
    return root / subject / session / "meg" / f"{subject}_{session}_task-{task}_markers.mrk"


def headshape_paths(root: Path, subject: str, session: str) -> Tuple[Path, Path]:
    hsp = root / subject / session / "meg" / f"{subject}_{session}_acq-HSP_headshape.pos"
    elp = root / subject / session / "meg" / f"{subject}_{session}_acq-ELP_headshape.pos"
    return hsp, elp


def load_events(events_tsv: Path) -> pd.DataFrame:
    return pd.read_csv(events_tsv, sep="\t")


def load_raw_kit(con_path: Path, mrk_path: Path, elp_path: Path, hsp_path: Path):
    if mne is None:
        raise RuntimeError("mne is required to load MEG data")
    # MNE expects .hsp/.elp/.txt; MEG-MASC uses .pos. Create temp .txt copies.
    def ensure_txt(path: Path) -> Path:
        if path.suffix in {".hsp", ".elp", ".txt", ".mat"}:
            return path
        if path.suffix == ".pos":
            txt_path = path.with_suffix(".txt")
            if not txt_path.exists():
                txt_path.write_bytes(path.read_bytes())
            return txt_path
        return path

    elp_path = ensure_txt(elp_path)
    hsp_path = ensure_txt(hsp_path)
    raw = mne.io.read_raw_kit(
        con_path,
        mrk=mrk_path,
        elp=elp_path,
        hsp=hsp_path,
        preload=True,
    )
    return raw


def extract_brain_window(raw, onset_sec: float, tmin: float, tmax: float) -> np.ndarray:
    """Extract brain window [onset+tmin, onset+tmax] in seconds."""
    start = onset_sec + tmin
    stop = onset_sec + tmax
    data, _ = raw[:, raw.time_as_index([start, stop])]
    return data.T  # shape [T, D]
