# PenMamba: IMU-pen handwriting recognition

Work in progress. This repository builds on the official code of [REWI](https://github.com/jindongli24/REWI) (CNN encoder + BiLSTM decoder, CTC loss) for word recognition from an IMU pen. The training code is the official one with small changes. All run settings are in `config.py`.

Dataset: OnHW-words500, right-handed, writer-independent split, 5 folds. 13 channels at 100 Hz: front accelerometer (3), rear accelerometer (3), gyroscope (3), magnetometer (3), force (1).

## Our results

All numbers are CER (%) of the last epoch on the validation folds, lower is better. They are our own runs with seed 42 on a Kaggle T4. Training is deterministic: the same seed gives the same result.

| run | what it is | f0 | f1 | f2 | f3 | f4 | mean CER | mean WER | params |
| --- | ---------- | --- | --- | --- | --- | --- | -------- | -------- | ------ |
| E1 baseline | REWI-Small, official code | 20.34 | 4.31 | 6.60 | 17.40 | 4.93 | 10.72 | 25.48 | 0.54M |
| E2 | BiMamba + 1-layer BiLSTM decoder | 21.15 | 5.67 | 8.07 | 18.53 | 5.64 | 11.81 | 29.71 | 0.39M |

E2 is worse than the baseline on all 5 folds.

Fold 0 only (other folds not run yet):

| run | what it is | CER | WER |
| --- | ---------- | --- | --- |
| E1 baseline | REWI-Small | 20.34 | 39.57 |
| E3 | sensor-group experts, learned gate | 19.91 | 38.64 |
| E3 control | sensor-group experts, equal weights | 19.39 | 38.10 |
| Base | REWI-Base (3.93M params), used as teacher | 16.46 | 28.89 |
| KD student | REWI-Small trained with the Base teacher | not run yet | not run yet |

## Setup

```bash
pip install jiwer Levenshtein loguru thop timm einops
```

## Data

Download OnHW-words500 (writer-independent) from the [Fraunhofer page](https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html), unzip it, then convert it to the REWI format. Paths are set in `rewi/config.py`.

```bash
python make_rewi_data.py
```

## Training

Run from the repository root. One fold per call.

```bash
python rewi/main.py --run test                       # 2-epoch speed test
python rewi/main.py --fold 0                         # E1 baseline
python rewi/main.py --fold 0 --arch-de mambalstm_s   # E2
python rewi/main.py --fold 0 --arch-en blconv_s_groups        # E3, learned gate
python rewi/main.py --fold 0 --arch-en blconv_s_groups_fixed  # E3 control
python rewi/main.py --fold 0 --arch-en blconv_b --arch-de bilstm_b  # Base teacher
python rewi/main.py --fold 0 --kd                    # Small student, distilled from Base
```

`--device cuda:1` picks the second GPU. Results go to `results/<run name>/<fold>/` (path set in `config.py`). The KD run needs the Base checkpoint of the same fold at `results/blconv_b_bilstm_b_wi/<fold>/checkpoints/299.pth`.

## Evaluation

Training already evaluates every 5 epochs. Results of all folds can be summarized with `evaluate.py`, as in the official repository.

## License

This project is released under the MIT license. Please see the `LICENSE` file for more information.

## Citiation

If you find this repository helpful, please consider citing:

```
@inproceedings{10.1007/978-3-032-13312-0_16,
  author    = {Li, Jindong
               and Hamann, Tim
               and Barth, Jens
               and K{\"a}mpf, Peter
               and Zanca, Dario
               and Eskofier, Bj{\"o}rn},
  editor    = {Durmaz Incel, {\"O}zlem
               and Qin, Jingwen
               and Bieber, Gerald
               and Kuijper, Arjan},
  title     = {Robust and Efficient Writer-Independent IMU-Based Handwriting Recognition},
  booktitle = {Sensor-Based Activity Recognition and Artificial Intelligence},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {261--286},
  isbn      = {978-3-032-13312-0}
}

```
