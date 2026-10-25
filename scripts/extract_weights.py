"""Extract weights from becruily_deux.ckpt into plain .npy files — numpy only, no torch.

torch zip ckpt format:
  model/data.pkl  — pickle; storages via persistent_load pid = ('storage', TypedStorageCls, key, device, numel)
  model/data/<n>  — raw storage bytes in creation order
"""
from __future__ import annotations

import zipfile
import pickle
import io
import json
import sys
from pathlib import Path

import numpy as np

TYPED = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64,
    "HalfStorage": np.float16, "BFloat16Storage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32,
    "ShortStorage": np.int16, "CharStorage": np.int8,
    "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class _OD(dict):
    """OrderedDict stand-in that tolerates attribute writes."""
    pass


def _storage_cls(name: str):
    t = type(name, (), {})
    t.__name__ = name
    return t


def _rebuild(st, storage_offset, size, stride, *a, **k):
    size = tuple(int(s) for s in size)
    stride = tuple(int(s) for s in stride)
    if not isinstance(st, np.ndarray):
        return np.zeros(size, dtype=np.float16)  # scan pass
    contiguous = stride[-1] == 1 and all(
        stride[i] == stride[i + 1] * size[i + 1] for i in range(len(size) - 1)
    )
    if contiguous or len(size) <= 1:
        n = int(np.prod(size))
        return np.ascontiguousarray(st[storage_offset: storage_offset + n].reshape(size))
    return np.lib.stride_tricks.as_strided(
        st[storage_offset:], shape=size,
        strides=tuple(s * st.itemsize for s in stride),
    ).copy()


class _NulStore:
    def __init__(self, key, numel):
        self.key, self.numel = key, numel


def load_ckpt_standalone(path: str) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as z:
        data_bytes = z.read("model/data.pkl")

        # ---- pass 1: capture storages in creation order -------------------
        captured: list = []

        class _Scan(pickle.Unpickler):
            def persistent_load(self, pid):
                _, cls, key, device, numel = pid
                name = getattr(cls, "__name__", "HalfStorage")
                captured.append((np.dtype(TYPED[name]), str(key), int(numel)))
                return _NulStore(key, numel)

            def find_class(self, module, name):
                if module.startswith("torch") and "Storage" in name:
                    return _storage_cls(name)
                if module.startswith("torch") and name in (
                    "_rebuild_tensor_v2", "_rebuild_tensor"
                ):
                    return _rebuild
                if module == "collections" and name == "OrderedDict":
                    return _OD
                class _D:
                    def __init__(self, *a, **k): pass
                    def __call__(self, *a, **k): return None
                    def __getattr__(self, n): return _D()
                    def __setattr__(self, n, v): pass
                return _D

        _Scan(io.BytesIO(data_bytes)).load()

        members = sorted(
            (n for n in z.namelist() if n.startswith("model/data/")),
            key=lambda n: int(n.rsplit("/", 1)[1]),
        )
        if len(members) != len(captured):
            raise RuntimeError(
                f"storage mismatch: {len(members)} data files vs {len(captured)} pids"
            )
        storage: dict[str, np.ndarray] = {}
        for member, (dt, key, numel) in zip(members, captured):
            raw = z.read(member)
            storage[key] = np.frombuffer(raw[: numel * dt.itemsize], dtype=dt)

    # ---- pass 2: rebuild real tensors -----------------------------------
    class _Load(pickle.Unpickler):
        def persistent_load(self, pid):
            _, cls, key, device, numel = pid
            return storage[str(key)]

        find_class = _Scan.find_class

    obj = _Load(io.BytesIO(data_bytes)).load()

    # torch ckpt: OrderedDict{name: Tensor}; possibly nested under 'state_dict'
    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj)
    else:
        sd = {}
    return {k: v for k, v in sd.items() if isinstance(v, np.ndarray)}


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        "/Volumes/nfs/data/Downloads/mel-band-roformer-deux/becruily_deux.ckpt"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "weights")
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = load_ckpt_standalone(ckpt)
    print(f"loaded {len(sd)} tensors")

    manifest = {}
    for name, arr in sd.items():
        rel = name.replace("model.", "", 1) if name.startswith("model.") else name
        rel = rel.replace(".", "/") + ".npy"
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
        manifest[name] = rel

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(a.nbytes for a in sd.values())
    print(f"wrote {len(manifest)} files, {total/1e6:.1f} MB -> {out_dir}")

    prefixes: dict[str, int] = {}
    for name, arr in sd.items():
        top = ".".join(name.split(".")[:3])
        prefixes.setdefault(top, 0)
        prefixes[top] += arr.nbytes
    for k, v in sorted(prefixes.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {v/1e6:9.2f} MB  {k}")


if __name__ == "__main__":
    main()
