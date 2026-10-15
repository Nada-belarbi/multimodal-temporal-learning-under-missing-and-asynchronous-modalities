import hashlib, json, math, os, shutil, tempfile
from pathlib import Path
import numpy as np

def to_jsonable(x):
    '''Recursively convert numpy types; NaN/Infinity become null (never serialised as numbers).'''
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x

def canonical_json(obj):
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf8')

def canonical_hash(obj):
    return hashlib.sha256(canonical_json(obj)).hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def dump_json(path, obj):
    Path(path).write_text(json.dumps(to_jsonable(obj), sort_keys=True, indent=2, allow_nan=False))

class AtomicDir:
    '''Write into a temp sibling folder; rename on success; never overwrite a completed result.'''
    def __init__(self, final):
        self.final = Path(final)
    def __enter__(self):
        if self.final.exists():
            raise FileExistsError(f'{self.final} already exists; completed results are never overwritten')
        self.final.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix=self.final.name + '.tmp-', dir=self.final.parent))
        return self.tmp
    def __exit__(self, et, ev, tb):
        if et is None:
            os.replace(self.tmp, self.final)
        else:
            shutil.rmtree(self.tmp, ignore_errors=True)
        return False
