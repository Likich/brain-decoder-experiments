#!/usr/bin/env python3
"""
Download OSF project ag3kj (MEG-MASC) without osfclient auth.

Uses OSF API to recursively list files and download via public links.

Example:
  python3 scripts/download_osf_ag3kj.py --out data/meg_masc --workers 4
  python3 scripts/download_osf_ag3kj.py --out data/meg_masc --include sub-01 --workers 2
"""

import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "https://api.osf.io/v2/nodes/ag3kj/files/osfstorage/"


def fetch_json(url: str):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def iter_folder(url: str):
    while url:
        data = fetch_json(url)
        for item in data.get("data", []):
            yield item
        url = data.get("links", {}).get("next")


def iter_files():
    stack = [ROOT]
    while stack:
        url = stack.pop()
        for item in iter_folder(url):
            kind = item["attributes"]["kind"]
            if kind == "folder":
                sub = item["relationships"]["files"]["links"]["related"]["href"]
                stack.append(sub)
            elif kind == "file":
                yield item


def download_one(
    url: str,
    path: Path,
    overwrite: bool,
    retries: int,
    timeout: int,
    user_agent: str,
    sleep_base: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return "skip", str(path)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(path)
            return "ok", str(path)
        except urllib.error.HTTPError as e:
            # retry on common transient errors
            if e.code in {403, 429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(sleep_base * (2**attempt))
                continue
            return "error", f"{path} (HTTP {e.code})"
        except Exception as e:  # noqa: BLE001
            if attempt < retries:
                time.sleep(sleep_base * (2**attempt))
                continue
            return "error", f"{path} ({e})"
    return "error", f"{path} (retries exhausted)"


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="Output root directory")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--include", type=str, default=None, help="Only download paths containing this substring")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--sleep_base", type=float, default=1.0)
    ap.add_argument("--user_agent", type=str, default="osf-downloader/1.0")
    args = ap.parse_args()

    items = []
    for item in iter_files():
        mat_path = item["attributes"]["materialized_path"].lstrip("/")
        if args.include and args.include not in mat_path:
            continue
        download = item["links"].get("download")
        if not download:
            continue
        local_path = args.out / mat_path
        items.append((download, local_path))
        if args.max_files and len(items) >= args.max_files:
            break

    if not items:
        raise SystemExit("No files matched.")

    print(f"Found {len(items)} files to download")

    ok = skip = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                download_one,
                url,
                path,
                args.overwrite,
                args.retries,
                args.timeout,
                args.user_agent,
                args.sleep_base,
            )
            for url, path in items
        ]
        for fut in as_completed(futures):
            status, p = fut.result()
            if status == "ok":
                ok += 1
                print(f"Downloaded {p}")
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"Failed {p}")
    print(f"Done. ok={ok} skip={skip} err={err}")


if __name__ == "__main__":
    main()
