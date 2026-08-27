#!/usr/bin/env python3
"""Pull representative weight slices out of a Llama safetensors checkpoint.

Runs on the machine that holds the checkpoint. **Standard library only** -- no
numpy, no torch, no safetensors package -- so it works on a bare Python install.

Why slices: the full 7B checkpoint is 13GB and SeedLM's exhaustive 65535-seed
search runs at ~4ms/block, so compressing all 6.5B linear weights would take
weeks. Reconstruction-error statistics converge long before that. What matters
is that the slices are *contiguous rows*: SeedLM blocks are C contiguous
elements in row-major order, so sampling whole rows preserves the exact block
structure the method would see on the real tensor. Sampling individual weights
would not.

Output is itself a valid safetensors file, readable with the safetensors package
or by this script's own parser.

    python3 extract_llama.py /path/to/llama2-7b /path/to/llama2_slices.safetensors
"""

import json
import os
import struct
import sys

# 7 projection types x a spread of depths.
PROJECTIONS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
DEPTH_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]

TARGET_ELEMENTS = 1_000_000   # per tensor, ~2MB in fp16
N_CHUNKS = 4                  # spread the rows over the tensor, keep each contiguous
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}


def read_header(path):
    """safetensors layout: u64 header length, JSON header, then the raw buffer."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def discover(model_dir):
    """Map tensor name -> shard path, from the index if present."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            wmap = json.load(f)["weight_map"]
        return {k: os.path.join(model_dir, v) for k, v in wmap.items()}
    mapping = {}
    for fn in sorted(os.listdir(model_dir)):
        if fn.endswith(".safetensors"):
            p = os.path.join(model_dir, fn)
            for name in read_header(p)[0]:
                if name != "__metadata__":
                    mapping.setdefault(name, p)
    return mapping


def n_layers(mapping):
    best = -1
    for name in mapping:
        parts = name.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
            best = max(best, int(parts[2]))
    return best + 1


def extract(model_dir, out_path):
    mapping = discover(model_dir)
    if not mapping:
        raise SystemExit(f"no safetensors tensors found under {model_dir}")
    nl = n_layers(mapping)
    print(f"found {len(mapping)} tensors across {nl} layers")

    layers = sorted({min(nl - 1, int(round(f * (nl - 1)))) for f in DEPTH_FRACTIONS})
    wanted = [f"model.layers.{L}.{p}.weight" for L in layers for p in PROJECTIONS]

    headers = {}                       # shard path -> (header, data_start)
    out_tensors, blobs, cursor = {}, [], 0
    skipped = []

    for name in wanted:
        shard = mapping.get(name)
        if shard is None or not os.path.exists(shard):
            skipped.append(name)
            continue
        if shard not in headers:
            headers[shard] = read_header(shard)
        header, data_start = headers[shard]
        meta = header[name]
        dtype, shape = meta["dtype"], meta["shape"]
        if len(shape) != 2:
            skipped.append(name)
            continue

        rows, cols = shape
        esz = DTYPE_BYTES[dtype]
        row_bytes = cols * esz
        want_rows = min(rows, max(N_CHUNKS, TARGET_ELEMENTS // cols))
        per_chunk = max(1, want_rows // N_CHUNKS)
        # Evenly spaced chunk starts; each chunk is a contiguous row range.
        starts = [min(rows - per_chunk, (i * rows) // N_CHUNKS) for i in range(N_CHUNKS)]

        base = data_start + meta["data_offsets"][0]
        got = 0
        with open(shard, "rb") as f:
            for s in starts:
                f.seek(base + s * row_bytes)
                want = per_chunk * row_bytes
                buf = f.read(want)
                if len(buf) != want:
                    # A short read here means the shard is incomplete -- almost
                    # always a download that ran out of disk.  Fail loudly: a
                    # partial slice would still parse and still produce
                    # plausible-looking numbers.
                    raise SystemExit(
                        f"\n{os.path.basename(shard)} is truncated: reading "
                        f"{name} rows {s}..{s+per_chunk} wanted {want:,} bytes, "
                        f"got {len(buf):,}.\nRe-download that shard "
                        f"(run with --check to see which ones are short)."
                    )
                blobs.append(buf)
                got += per_chunk

        nbytes = got * row_bytes
        out_tensors[name] = {
            "dtype": dtype,
            "shape": [got, cols],
            "data_offsets": [cursor, cursor + nbytes],
        }
        cursor += nbytes
        print(f"  {name:<48} {dtype:>4}  {rows}x{cols} -> {got}x{cols}")

    if skipped:
        print(f"skipped {len(skipped)} tensors (shard absent): "
              f"{', '.join(skipped[:3])}{' ...' if len(skipped) > 3 else ''}")
    if not out_tensors:
        raise SystemExit("nothing extracted")

    out_tensors["__metadata__"] = {
        "source": os.path.abspath(model_dir),
        "note": "contiguous row slices; block structure preserved",
    }
    blob = json.dumps(out_tensors).encode()
    pad = (-len(blob)) % 8               # safetensors wants an 8-byte aligned buffer
    blob += b" " * pad
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for b in blobs:
            f.write(b)

    mb = os.path.getsize(out_path) / 1e6
    print(f"\nwrote {out_path}  ({len(out_tensors) - 1} tensors, {mb:.1f} MB)")


def plan(model_dir):
    """Print the shards actually needed, from the index alone.

    Large checkpoints spread layers over dozens of shards, and this samples only
    5 depths, so most shards are never touched.  Download
    ``model.safetensors.index.json`` first, run this, and fetch only what it
    lists -- for Llama-3-70B that is roughly 23GB out of 141GB.
    """
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index):
        raise SystemExit(f"need {index} (download just that file first)")
    with open(index) as f:
        wmap = json.load(f)["weight_map"]

    nl = max(int(k.split(".")[2]) for k in wmap
             if k.startswith("model.layers.")) + 1
    layers = sorted({min(nl - 1, int(round(f * (nl - 1)))) for f in DEPTH_FRACTIONS})

    need, missing = {}, []
    for L in layers:
        for p in PROJECTIONS:
            name = f"model.layers.{L}.{p}.weight"
            if name in wmap:
                need.setdefault(wmap[name], []).append(f"L{L}.{p}")
            else:
                missing.append(name)

    print(f"{nl} layers; sampling depths {layers}")
    print(f"{len(wmap)} tensors across {len(set(wmap.values()))} shards; "
          f"{len(need)} shard(s) needed\n")
    for shard in sorted(need):
        print(f"  {shard:<44} {len(need[shard])} tensors")
    if missing:
        print(f"\n  !! {len(missing)} expected tensors absent from the index: "
              f"{missing[:2]}")
    print("\nhf download <repo> \\")
    for shard in sorted(need):
        print(f"    {shard} \\")
    print("    model.safetensors.index.json config.json \\\n    --local-dir <dir>")


def check(model_dir):
    """Report any shard whose bytes on disk fall short of its own header.

    safetensors records every tensor's byte range up front, so the file's
    required length is knowable without reading the payload.  A download that
    ran out of disk leaves a file that is still valid-looking and still far
    larger than any size threshold -- this is the check that catches it.
    """
    import glob
    paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not paths:
        raise SystemExit(f"no safetensors files in {model_dir}")
    bad = 0
    for p in paths:
        try:
            header, data_start = read_header(p)
        except Exception as exc:
            print(f"  {os.path.basename(p):<44} UNREADABLE HEADER ({exc})")
            bad += 1
            continue
        t = {k: v for k, v in header.items() if k != "__metadata__"}
        need = data_start + max(v["data_offsets"][1] for v in t.values())
        have = os.path.getsize(p)
        ok = have >= need
        bad += not ok
        print(f"  {os.path.basename(p):<44} {have:>15,} / {need:>15,}  "
              + ("ok" if ok else f"SHORT by {need - have:,}"))
    print(f"\n{len(paths) - bad}/{len(paths)} complete")
    if bad:
        print("Re-download the short ones; nothing extracted from them is usable.")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        check(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "--plan":
        plan(sys.argv[2])
    elif len(sys.argv) == 3:
        extract(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(__doc__)
