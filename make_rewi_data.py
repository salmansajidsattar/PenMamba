"""Convert OnHW pickles to the CSV + JSON layout the official REWI code reads.

Run: python make_rewi_data.py --data-root DIR --out DIR [--folds 0 1] [--limit 300]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rewi"))
import config  # noqa: E402

import data


def save_sample(seq, path):
    """Write one (T, 13) signal as ';' separated text. Values are whole numbers, so '%d' loses nothing."""
    if not np.array_equal(seq, np.round(seq)):
        raise ValueError("signal has non-integer values; change the format")
    np.savetxt(path, seq, fmt="%d", delimiter=";")


def pick_indexes(n, limit):
    """All indexes, or `limit` indexes spread over the whole part (for a quick test)."""
    if not limit or limit >= n:
        return range(n)
    return np.linspace(0, n - 1, limit).astype(int)


def convert_part(split, fold, part, out_dir, limit):
    """Write the kept samples of one fold and part; return the annotation list."""
    x, y, ids = data.load_fold(split, fold, part)
    folder = os.path.join("data", str(fold), part)
    os.makedirs(os.path.join(out_dir, folder), exist_ok=True)
    annos = []
    for i in pick_indexes(len(x), limit):
        if not data.keep_sample(x[i]):  # same filter as the REWI notebook
            continue
        name = os.path.join(folder, str(len(annos)).zfill(8) + ".csv")
        save_sample(x[i], os.path.join(out_dir, name))
        annos.append({"filename": name, "id": len(annos), "label": data.ids_to_text(y[i]),
                      "id_writer": int(ids[i]), "rate_sample_orig": 100})
    print(f"fold {fold} {part}: kept {len(annos)} of {len(x)}")
    return annos


def write_json(split, folds, part, out_dir, limit):
    """Write train.json or val.json with the annotations of all asked folds."""
    annos = {str(f): convert_part(split, f, part, out_dir, limit) for f in folds}
    info = {"num_channel": 13, "rate_sample_target": 100, "num_fold": 5,
            "writer_independent": split == "wi"}
    with open(os.path.join(out_dir, part + ".json"), "w") as f:
        json.dump({"info": info, "categories": sorted(data.CHARS), "annotations": annos}, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=config.RAW_DIR)
    p.add_argument("--out", default=config.CONVERTED_DIR)
    p.add_argument("--split", default=config.SPLIT)
    p.add_argument("--folds", type=int, nargs="+", default=list(range(config.NUM_FOLD)))
    p.add_argument("--limit", type=int, default=0, help="samples per part, for a quick test")
    a = p.parse_args()
    data.DATA_ROOT = a.data_root
    for part in ("train", "val"):
        write_json(a.split, a.folds, part, a.out, a.limit)
