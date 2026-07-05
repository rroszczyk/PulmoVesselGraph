"""Central cache manager for the pulmonary vascular tree pipeline.

Cache layout:
    cache/
        index.json                          - global index
        <stage_name>/
            <sha256>.npz                    - numpy arrays (compressed)
            <sha256>.pkl                    - Python objects (NetworkX graphs, dicts)
            <sha256>.meta.json              - metadata (timestamp, shapes, params)
"""
import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CacheManager:
    """SHA256-keyed, multi-format cache for pipeline stages.

    Args:
        cache_dir: Root directory for all cached data.
    """

    def __init__(self, cache_dir: str = "cache") -> None:
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        self._index: Dict[str, Any] = self._load_index()


    def get(self, stage: str, params: dict) -> Optional[Dict[str, Any]]:
        """Load cached result for *stage* with *params*.

        Returns:
            dict of arrays/objects, or None if cache miss.
        """
        key = self._make_key(stage, params)
        stage_dir = self.root / stage
        npz_path  = stage_dir / f"{key}.npz"
        pkl_path  = stage_dir / f"{key}.pkl"
        meta_path = stage_dir / f"{key}.meta.json"

        if not meta_path.exists():
            logger.debug("Cache MISS  [%s] key=%s", stage, key[:12])
            return None

        logger.info("Cache HIT   [%s] key=%s", stage, key[:12])
        result: Dict[str, Any] = {}

        # Load numpy arrays
        if npz_path.exists():
            with np.load(str(npz_path), allow_pickle=False) as npz:
                for k in npz.files:
                    result[k] = npz[k]

        # Load pickled objects (graphs, dicts, …)
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                pkl_data: dict = pickle.load(f)
            result.update(pkl_data)

        return result

    def put(
        self,
        stage: str,
        params: dict,
        data: Dict[str, Any],
        numpy_keys: Optional[List[str]] = None,
    ) -> None:
        """Persist *data* for *stage* with *params*.

        Args:
            stage:      Stage name (used as subdirectory).
            params:     Params dict (used to build the cache key).
            data:       Dict of arrays and/or objects to store.
            numpy_keys: Keys whose values should be saved as .npz
                        (must be np.ndarray).  All other keys go to .pkl.
                        If None, arrays are detected automatically.
        """
        key = self._make_key(stage, params)
        stage_dir = self.root / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        npz_path  = stage_dir / f"{key}.npz"
        pkl_path  = stage_dir / f"{key}.pkl"
        meta_path = stage_dir / f"{key}.meta.json"

        # Split data into numpy vs pickle
        if numpy_keys is None:
            numpy_keys = [k for k, v in data.items() if isinstance(v, np.ndarray)]
        pickle_keys = [k for k in data if k not in numpy_keys]

        # Save numpy arrays
        if numpy_keys:
            arrays = {k: data[k] for k in numpy_keys if k in data}
            np.savez_compressed(str(npz_path), **arrays)
            logger.debug("Saved .npz  [%s] keys=%s", stage, list(arrays.keys()))

        # Save pickled objects
        if pickle_keys:
            pkl_data = {k: data[k] for k in pickle_keys if k in data}
            with open(pkl_path, "wb") as f:
                pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Saved .pkl  [%s] keys=%s", stage, list(pkl_data.keys()))

        # Write metadata
        meta = {
            "stage":      stage,
            "key":        key,
            "timestamp":  time.time(),
            "iso_ts":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params":     params,
            "numpy_keys": numpy_keys,
            "pickle_keys": pickle_keys,
            "shapes": {
                k: list(data[k].shape)
                for k in numpy_keys
                if k in data and hasattr(data[k], "shape")
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        # Update global index
        self._index.setdefault(stage, {})[key] = {
            "ts":     meta["iso_ts"],
            "params": params,
        }
        self._save_index()
        logger.info("Cache WRITE [%s] key=%s", stage, key[:12])

    def invalidate(self, stage: Optional[str] = None) -> None:
        """Delete cached files.

        Args:
            stage: If given, only delete that stage's cache.
                   If None, delete everything.
        """
        import shutil

        if stage is None:
            for child in self.root.iterdir():
                if child.is_dir() and child.name != "previews":
                    shutil.rmtree(child)
            self._index = {}
            if self._index_path.exists():
                self._index_path.unlink()
            logger.info("Cache cleared (all stages)")
        else:
            stage_dir = self.root / stage
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            self._index.pop(stage, None)
            self._save_index()
            logger.info("Cache cleared [%s]", stage)

    def print_status(self) -> None:
        """Print a summary of all cached stages to the logger."""
        lines = ["=" * 60, "Cache status:"]
        total_bytes = 0
        for stage, entries in sorted(self._index.items()):
            stage_dir = self.root / stage
            stage_bytes = sum(
                f.stat().st_size
                for f in stage_dir.iterdir()
                if f.is_file()
            ) if stage_dir.exists() else 0
            total_bytes += stage_bytes
            lines.append(
                f"  [{stage}]  {len(entries)} entr(ies)  "
                f"{stage_bytes / 1024 / 1024:.1f} MB"
            )
        lines.append(f"  TOTAL: {total_bytes / 1024 / 1024:.1f} MB")
        lines.append("=" * 60)
        logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(stage: str, params: dict) -> str:
        """Deterministic SHA-256 key from stage name + params."""
        payload = json.dumps(
            {"stage": stage, "params": params}, sort_keys=True, default=str
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                with open(self._index_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, default=str)
