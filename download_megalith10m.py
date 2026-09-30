#!/usr/bin/env python3
"""
Resumable downloader for madebyollin/megalith-10m.

Megalith-10m does NOT contain image files - it's ~9.58M rows of Flickr
image URLs (columns: url, url_highres, url_source), stored as 10 parquet
shards on the Hugging Face Hub. This script:

  1. Downloads the parquet shards from the Hub (via huggingface_hub, which
     caches to disk -- if a shard download is interrupted, re-running just
     resumes/re-fetches that shard, already-cached shards aren't re-downloaded).
  2. Streams through the rows and downloads each linked image from Flickr
     with a thread pool.
  3. Tracks progress in a small SQLite state file so that if the process is
     killed (Ctrl-C, crash, power loss, whatever), re-running the same
     command picks up where it left off instead of starting over -- rows
     already marked "done" are skipped, and any image file that already
     exists on disk is also skipped as an extra safety net.

Install dependencies:
    pip install huggingface_hub pyarrow pandas requests tqdm

Usage:
    python download_megalith10m.py --output-dir ./megalith_images --limit 50000
    # re-run the exact same command any time to resume
    python download_megalith10m.py --output-dir ./megalith_images --limit 50000

Useful options:
    --workers 16          number of parallel download threads (default 1)
    --limit N              stop after N successfully downloaded images (omit for "all")
    --shards 0 1 2         only process specific shard indices (0-9), default: all 10
    --highres              download the higher-resolution "url_highres" variant instead of "url"
    --min-bytes 2048       treat responses smaller than this as failures (placeholder/error pages)
"""

import argparse
import io
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path

import requests
from tqdm import tqdm

SHARD_COUNT = 10
REPO_ID = "madebyollin/megalith-10m"
REPO_TYPE = "dataset"


def shard_filename(i: int) -> str:
    return f"data/megalith-{i+1:05d}-of-{SHARD_COUNT:05d}.parquet"


def download_shards(cache_dir: Path, shard_indices):
    """Download (or reuse cached) parquet shards. Returns list of local paths."""
    from huggingface_hub import hf_hub_download

    paths = []
    for i in shard_indices:
        fname = shard_filename(i)
        print(f"[shards] fetching {fname} (cached if already present)...")
        # hf_hub_download resumes partial downloads itself and skips fully
        # cached files, so this is safe to re-run after an interruption.
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=fname,
            local_dir=str(cache_dir / "parquet_cache"),
        )
        paths.append(local_path)
    return paths


class ProgressDB:
    """Tiny SQLite-backed record of which rows have been handled."""

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS rows (
                   photo_id TEXT PRIMARY KEY,
                   status   TEXT NOT NULL,       -- 'done' or 'failed'
                   ts       REAL NOT NULL
               )"""
        )
        self.conn.commit()
        self._lock_conn = self.conn

    def already_done(self, photo_id: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM rows WHERE photo_id=? AND status='done'", (photo_id,)
        )
        return cur.fetchone() is not None

    def mark(self, photo_id: str, status: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO rows (photo_id, status, ts) VALUES (?, ?, ?)",
            (photo_id, status, time.time()),
        )

    def commit(self):
        self.conn.commit()

    def count_done(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM rows WHERE status='done'")
        return cur.fetchone()[0]


def photo_id_from_source(url_source: str, fallback: str) -> str:
    # url_source looks like: https://www.flickr.com/photos/<user>/<photo_id>
    try:
        return url_source.rstrip("/").rsplit("/", 1)[-1]
    except Exception:
        return fallback


def out_path_for(images_dir: Path, photo_id: str, url: str) -> Path:
    ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
    if len(ext) > 5:  # sanity guard against weird urls
        ext = ".jpg"
    # bucket into subfolders so we don't put millions of files in one dir
    bucket = photo_id[-2:] if len(photo_id) >= 2 else "00"
    d = images_dir / bucket
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{photo_id}{ext}"


def download_one(session, url, dest: Path, min_bytes: int, timeout: float) -> bool:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return True
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    try:
        resp = session.get(url, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return False
        buf = io.BytesIO()
        for chunk in resp.iter_content(chunk_size=65536):
            buf.write(chunk)
        data = buf.getvalue()
        if len(data) < min_bytes:
            return False
        tmp_dest.write_bytes(data)
        tmp_dest.replace(dest)  # atomic rename -> no partial files left as final output
        return True
    except requests.RequestException:
        return False
    finally:
        if tmp_dest.exists():
            try:
                tmp_dest.unlink()
            except OSError:
                pass


def iter_rows(parquet_paths, url_column):
    import pyarrow.parquet as pq

    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            batch_size=4096, columns=[url_column, "url_source"]
        ):
            urls = batch.column(url_column).to_pylist()
            sources = batch.column("url_source").to_pylist()
            for url, src in zip(urls, sources):
                if not url:
                    continue
                pid = photo_id_from_source(src or "", fallback=str(hash(url)))
                yield pid, url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="./megalith_images", help="Where images + state are stored")
    ap.add_argument("--workers", type=int, default=1, help="Parallel download threads")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N successful downloads (default: no limit)")
    ap.add_argument("--shards", type=int, nargs="*", default=list(range(SHARD_COUNT)), help="Which shard indices (0-9) to process")
    ap.add_argument("--highres", action="store_true", help="Download url_highres instead of url")
    ap.add_argument("--min-bytes", type=int, default=2048, help="Minimum response size to count as a valid image")
    ap.add_argument("--timeout", type=float, default=15.0, help="Per-request timeout (seconds)")
    ap.add_argument("--user-agent", default="megalith-10m-downloader/1.0 (personal research use)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = ProgressDB(out_dir / "progress.sqlite3")
    print(f"[state] {db.count_done()} images already recorded as done in {out_dir/'progress.sqlite3'}")

    parquet_paths = download_shards(out_dir, args.shards)

    url_column = "url_highres" if args.highres else "url"

    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})

    successes = db.count_done()
    target = args.limit
    pool_holder = {}  # lets the KeyboardInterrupt handler reach the live executor

    def submit_and_run():
        nonlocal successes
        pool = ThreadPoolExecutor(max_workers=args.workers)
        pool_holder["pool"] = pool
        target_hit = False
        try:
            futures = {}
            pbar = tqdm(desc="downloading", unit="img", initial=successes, total=target)
            batch_since_commit = 0

            def target_reached():
                return target is not None and successes >= target

            def drain_completed():
                """Process any finished futures. Returns True if target reached."""
                nonlocal successes, batch_since_commit
                done_now = [fut for fut in list(futures.keys()) if fut.done()]
                for fut in done_now:
                    pid, dest = futures.pop(fut)
                    try:
                        ok = fut.result()
                    except Exception:
                        ok = False
                    db.mark(pid, "done" if ok else "failed")
                    batch_since_commit += 1
                    if ok:
                        successes += 1
                        pbar.update(1)
                    if batch_since_commit >= 200:
                        db.commit()
                        batch_since_commit = 0
                    if target_reached():
                        return True
                return False

            for pid, url in iter_rows(parquet_paths, url_column):
                if target_reached():
                    target_hit = True
                    break
                if db.already_done(pid):
                    continue
                dest = out_path_for(images_dir, pid, url)
                if dest.exists() and dest.stat().st_size >= args.min_bytes:
                    db.mark(pid, "done")
                    successes += 1
                    pbar.update(1)
                    if target_reached():
                        target_hit = True
                        break
                    continue

                fut = pool.submit(download_one, session, url, dest, args.min_bytes, args.timeout)
                futures[fut] = (pid, dest)

                # keep `successes` current every cycle (not just at a big backlog)
                # so we notice hitting --limit promptly instead of overshooting it
                if drain_completed():
                    target_hit = True
                    break
                if len(futures) >= args.workers * 2:
                    # backlog is full: wait for at least one slot to free up
                    wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                    if drain_completed():
                        target_hit = True
                        break

            if target_hit:
                # don't wait for the rest of the in-flight downloads to finish -
                # just drop them and stop, so --limit actually stops promptly
                db.commit()
                pbar.close()
                return

            # dataset exhausted before hitting --limit (or no --limit set):
            # wait for whatever's still in flight so nothing is lost
            for fut in as_completed(list(futures.keys())):
                pid, dest = futures.pop(fut)
                try:
                    ok = fut.result()
                except Exception:
                    ok = False
                db.mark(pid, "done" if ok else "failed")
                if ok:
                    successes += 1
                    pbar.update(1)

            db.commit()
            pbar.close()
        finally:
            # cancel_futures drops anything not yet started; wait=False means
            # we don't block on in-flight requests finishing their timeout
            pool.shutdown(wait=False, cancel_futures=True)

    try:
        submit_and_run()
    except KeyboardInterrupt:
        print("\n[interrupted] saving progress...")
        pool = pool_holder.get("pool")
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        db.commit()
        print(f"[interrupted] {successes} images recorded as done. Re-run the same command to resume.")
        os._exit(1)  # skip concurrent.futures' atexit thread-join, which would hang

    print(f"[done] {successes} images downloaded into {images_dir}")
    print(f"[state] progress DB: {out_dir/'progress.sqlite3'} (delete it to force a full re-download)")
    # hard-exit: any straggler threads (e.g. one still mid-request when the
    # limit was hit) are daemon-less by default and would otherwise block
    # interpreter shutdown until they finish on their own
    os._exit(0)


if __name__ == "__main__":
    main()