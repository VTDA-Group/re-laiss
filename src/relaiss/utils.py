import logging
import os
import sys
import warnings
from contextlib import contextmanager
import hashlib
import pickle
from pathlib import Path
import pandas as pd
import hashlib, json
import math
from typing import Optional

def _fmt_z(val: Optional[float]) -> str:
    """Right-aligned z column; '---' if missing."""
    if val is None or (isinstance(val, (float, int)) and math.isnan(val)):
        return "---"
    return f"{val:7.3f}"

def _fmt_txt(text: Optional[str], width: int) -> str:
    """Text column with fallback ‘---’."""
    return (text or "---").ljust(width)[:width]

def compress(obj) -> str:
    """Return a short hash of any JSON-serialisable object."""
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()

@contextmanager
def suppress_output():
    """Temporarily silence stdout, stderr, warnings and logging < CRITICAL."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        logging.disable(logging.CRITICAL)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
        finally:
            logging.disable(logging.NOTSET)
            sys.stdout, sys.stderr = old_stdout, old_stderr

def get_cache_dir():
    """Get the cache directory path, creating it if it doesn't exist."""
    cache_dir = Path.home() / '.relaiss' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def get_cache_key(data_hash, **kwargs):
    """Generate a cache key from data hash and parameters."""
    key_parts = [str(data_hash)]
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (list, tuple)):
            v = ','.join(sorted(str(x) for x in v))
        key_parts.append(f"{k}={v}")
    key_str = '_'.join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()

def compute_dataframe_hash(df):
    """Compute a hash of a pandas DataFrame."""
    return hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()

def cache_dataframe(df, cache_key, cache_dir=None):
    """Cache a DataFrame to disk."""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    cache_path = cache_dir / f"{cache_key}.pkl"
    df.to_pickle(str(cache_path))
    return cache_path

def load_cached_dataframe(cache_key, cache_dir=None):
    """Load a cached DataFrame from disk."""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    cache_path = cache_dir / f"{cache_key}.pkl"
    if cache_path.exists():
        return pd.read_pickle(str(cache_path))
    return None

def clear_cache(cache_dir=None):
    """Clear all cached files."""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    for f in cache_dir.glob("*.pkl"):
        f.unlink()
