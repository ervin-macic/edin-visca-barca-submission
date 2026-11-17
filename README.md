# Ball possession per frame: FC Barcelona – Athletic Club

Team Edin Visca Barca, More than a Hack 2027. This repository regenerates our possession file
from the provided match data. The match data itself is not included.

## Install

Python 3.12, CPU only:

    python3.12 -m venv .venv
    .venv/bin/pip install -r requirements.txt

## Run

    .venv/bin/python make_submission.py --data-dir /path/to/data --out submission.csv

`--data-dir` is the unpacked match data bundle (default: `$MTAH_DATA_DIR`, else `./data`). These
are the only files read; the videos are not used:

    tactical/BAR-ATH_tactical_atd.txt
    tactical/BAR-ATH_tactical_atd.xml
    tactical/BAR-ATH_tactical_homography.csv
    tactical/BAR-ATH_tactical_smart_tagging.xml
    broadcast/BAR-ATH_broadcast_atd.txt
    broadcast/BAR-ATH_broadcast_atd.xml

A run takes about a minute and 1.5 GB of memory. The output is byte-identical to our submitted
file:

    $ sha256sum submission.csv
    b55bcc691873d04a852e60001d8aba72cab082b6abd34fe70976a4b50c641782  submission.csv

## Contents

- `make_submission.py`: the pipeline, step by step
- `possession/`: reading and cleaning the tracking, ball fusion, features, models, decoding
- `models/`: the trained models (LightGBM, two PyTorch networks), trained on other matches only
- `training/`: the code that trains them

## Training (optional)

The models in `models/` are trained on 10 other matches: the DFL open data (7 matches, with
per-frame ball-status and possession flags) and the Metrica Sports sample data (3 matches, labels
derived from its events). To train them again (writes into `models/`):

    .venv/bin/python -m training.download          # the two datasets, about 4.5 GB
    .venv/bin/python -m training.dfl               # convert to frames and labels
    .venv/bin/python -m training.metrica
    .venv/bin/python -m training.train_trees       # lightgbm_*.txt, about 15 minutes
    .venv/bin/python -m training.train_networks --data-dir /path/to/data    # mlp48.pt, tcn24.pt

The match data is read only by `train_networks`, to measure without labels how much of each ball
gap the fused ball fills (`training/fill.py`); the networks are trained on the other matches with
that fill simulated. The networks train on the CPU with 4 threads; on another machine the weights
can differ in the last digits.
