"""Load OnHW-words500 folds. Numpy only. Run `python data.py wi 0` for the E0 summary."""
import os
import pickle
import sys

import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SPLITS = {"wi": "Words500_indep_02", "wd": "Words500_dep_02"}
PAD_LABEL = 59  # pads every word to 19 entries (a space in the REWI notebook)
MAX_LEN = 1024  # REWI drops longer samples
# Label id = index in this string (taken from the REWI onhw.ipynb notebook).
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß"


class _SafeUnpickler(pickle.Unpickler):
    """Allow only numpy and plain Python types, because pickle can run code."""

    def find_class(self, module, name):
        if module.split(".")[0] in ("numpy", "builtins", "collections"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def safe_load(path):
    """Load one pickle file with the safe unpickler."""
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


def load_fold(split, fold, part):
    """Return (signals, labels, writer_ids) for part 'train' or 'val'."""
    d = os.path.join(DATA_ROOT, SPLITS[split], str(fold))
    x = safe_load(os.path.join(d, f"all_x_dat_{part}_imu.pkl"))  # list of (T, 13)
    y = safe_load(os.path.join(d, f"all_{part}_gt.pkl"))  # list of 19 label ids
    ids = safe_load(os.path.join(d, f"{part}_ids.pkl"))  # writer id per sample
    return x, y, ids


def strip_padding(label):
    """Drop PAD_LABEL entries; what is left is the real word."""
    return [c for c in label if c != PAD_LABEL]


def ids_to_text(label):
    """Turn a padded list of label ids into the word as a string."""
    return "".join(CHARS[c] for c in label if c != PAD_LABEL)


def keep_sample(signal):
    """REWI keeps only samples that are not empty and not longer than MAX_LEN."""
    return 0 < len(signal) <= MAX_LEN


def describe_part(name, x, y, ids):
    """Print sizes, signal lengths and word lengths for one part."""
    lens = np.array([len(a) for a in x])
    words = [strip_padding(w) for w in y]
    wl = np.array([len(w) for w in words])
    print(f"[{name}] samples {len(x)}, writers {len(set(ids))}")
    print(f"  signal length min/mean/max {lens.min()}/{lens.mean():.0f}/{lens.max()}")
    print(f"  length 0: {(lens == 0).sum()}, under 8: {(lens < 8).sum()}, over 1024: {(lens > 1024).sum()}")
    print(f"  word length min/max {wl.min()}/{wl.max()}, empty words {(wl == 0).sum()}")
    print(f"  kept by REWI filter: {sum(keep_sample(a) for a in x)} of {len(x)}")
    print(f"  too short for CTC after 8x shrink: {(lens // 8 < wl).sum()}")
    return words


def summarize(split, fold):
    """E0 check: sizes, writer overlap, distinct words, train/val word overlap."""
    xt, yt, it = load_fold(split, fold, "train")
    xv, yv, iv = load_fold(split, fold, "val")
    wt = describe_part("train", xt, yt, it)
    wv = describe_part("val", xv, yv, iv)
    print("example words:", [ids_to_text(w) for w in yt[:6]])
    print(f"writer overlap train/val: {len(set(it) & set(iv))}")
    train_words = {tuple(w) for w in wt}
    val_words = [tuple(w) for w in wv]
    seen = np.mean([w in train_words for w in val_words])
    print(f"distinct words train {len(train_words)}, val {len(set(val_words))}")
    print(f"val samples whose word is also in train: {seen:.1%}")
    print(f"channels per sample: {sorted({a.shape[1] for a in xt})}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "wi"
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    summarize(split, fold)
