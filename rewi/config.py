"""All run settings in one place. main.py and make_rewi_data.py read their values from here.

    python main.py            -> real run   (run="e1")
    python main.py --run test -> 2-epoch speed test
"""
import os


# ---- where we run (paths change, nothing else) ----
if os.path.exists("/kaggle"):
    RAW_DIR = "/kaggle/temp/raw"                      # unzipped dataset
    CONVERTED_DIR = "/kaggle/temp/onhw_wi_word_rh"    # name must contain 'word' (REWI uses it for MACs)
    RESULTS_DIR = "/kaggle/working/results"
    TEST_WORK_DIR = "/kaggle/temp/test_run"
    NUM_WORKER = 2
else:  # Colab
    RAW_DIR = "/content/raw"
    CONVERTED_DIR = "/content/onhw_wi_word_rh"
    RESULTS_DIR = "/content/drive/MyDrive/onhw/results"
    TEST_WORK_DIR = "/content/test_run"
    NUM_WORKER = 2

# ---- experiment ----
SPLIT = "wi"            # wi = writer-independent, wd = writer-dependent
FOLD = 0
NUM_FOLD = 5
ARCH_EN = "blconv_s"    # REWI-Small encoder
ARCH_DE = "bilstm_s"    # REWI-Small decoder
DEVICE = "cuda"         # "cuda:0" / "cuda:1" to pick a GPU
MULTI_GPU = False       # True = split each batch over all GPUs (nn.DataParallel)


def run_name(arch_en=ARCH_EN, arch_de=ARCH_DE):
    """Results go to RESULTS_DIR/<run_name>/<fold>/. The baseline keeps its old name."""
    if arch_de == "bilstm_s":
        return f"{arch_en}_{SPLIT}"
    return f"{arch_en}_{arch_de}_{SPLIT}"


# ---- training (REWI-Small recipe) ----
EPOCH = 300
EPOCH_WARMUP = 30
LR = 0.001
BATCH_SIZE = 64
SEED = 42
FREQ_EVAL = 5
FREQ_SAVE = 5
FREQ_LOG = 300

# ---- speed test ----
TEST_EPOCH = 2
TEST_EPOCH_WARMUP = 1

# ---- data ----
NUM_CHANNEL = 13
# same order as the official REWI config ("" is the CTC blank)
CATEGORIES = [""] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜabcdefghijklmnopqrstuvwxyzäöüß")


def make_cfg(epoch=EPOCH, epoch_warmup=EPOCH_WARMUP, work_dir=None, fold=FOLD,
             arch_en=ARCH_EN, arch_de=ARCH_DE, device=DEVICE, multi_gpu=MULTI_GPU):
    """Return the settings dict in the form REWI's main.py expects."""
    if work_dir is None:
        work_dir = os.path.join(RESULTS_DIR, run_name(arch_en, arch_de))
    return {
        "arch_de": arch_de,
        "arch_en": arch_en,
        "aug": True,
        "cache": True,
        "categories": CATEGORIES,
        "checkpoint": None,
        "ctc_decoder": "best_path",
        "device": device,
        "dir_dataset": CONVERTED_DIR,
        "dir_work": work_dir,
        "epoch": epoch,
        "epoch_warmup": epoch_warmup,
        "freeze": False,
        "freq_eval": FREQ_EVAL,
        "freq_log": FREQ_LOG,
        "freq_save": FREQ_SAVE,
        "idx_fold": fold,
        "len_seq": 0,
        "lr": LR,
        "multi_gpu": multi_gpu,
        "num_channel": NUM_CHANNEL,
        "num_worker": NUM_WORKER,
        "seed": SEED,
        "size_batch": BATCH_SIZE,
        "test": False,
    }


def get_cfg(run="e1", fold=FOLD, arch_de=None, device=None, multi_gpu=None, arch_en=None):
    """run = "e1" (real run) or "test" (2-epoch speed test); fold = 0..4.
    arch_en, arch_de, device, multi_gpu: override the defaults above (None = keep default)."""
    extra = {
        "arch_en": arch_en or ARCH_EN,
        "arch_de": arch_de or ARCH_DE,
        "device": device or DEVICE,
        "multi_gpu": MULTI_GPU if multi_gpu is None else multi_gpu,
    }
    if run == "test":
        return make_cfg(TEST_EPOCH, TEST_EPOCH_WARMUP, TEST_WORK_DIR, fold, **extra)
    return make_cfg(fold=fold, **extra)
