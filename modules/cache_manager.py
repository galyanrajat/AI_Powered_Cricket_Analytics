import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any

CACHE_FILE = Path("logs/cache.json")

def _hash_files(files: List[Path]) -> str:
    """Return a combined hash of file contents and modified times."""
    hash_md5 = hashlib.md5()
    for f in files:
        if f.exists():
            hash_md5.update(str(f.stat().st_mtime).encode())
            with open(f, "rb") as file:
                hash_md5.update(file.read())
    return hash_md5.hexdigest()

def needs_update(step_name: str, files: List[Path], config: Dict[str, Any]) -> bool:
    """
    Check if the step needs to be re-run based on:
      - Missing output files
      - Changed input files
      - Changed config
    """
    if not CACHE_FILE.exists():
        return True

    cache = json.loads(CACHE_FILE.read_text())

    file_hash = _hash_files(files)
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

    prev = cache.get(step_name)
    if not prev:
        return True

    return prev["file_hash"] != file_hash or prev["config_hash"] != config_hash

def update_cache(step_name: str, files: List[Path], config: Dict[str, Any]):
    """Update cache record for this step."""
    cache = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())

    file_hash = _hash_files(files)
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

    cache[step_name] = {
        "file_hash": file_hash,
        "config_hash": config_hash
    }

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

