"""Write the possession file (one H / A / D / X label per tactical video frame) from the match data.

    python make_submission.py --data-dir path/to/data --out submission.csv

Steps:
  1. read the tactical tracking and clean its goalkeeper slots (possession/cleaning.py);
  2. build the fused ball: the tactical ball with gaps filled from the broadcast tracking
     (possession/ball.py);
  3. per frame, probabilities of H / A / D and of a possession change from three models trained on
     other matches, averaged in equal thirds (possession/models.py);
  4. Viterbi decoding, then dead ball before each half's kick-off (possession/decode.py);
  5. write `frame,label`, with X outside the two graded halves.
"""

import argparse
import os

import numpy as np

from possession import ball, cleaning, decode, models
from possession.data import LETTERS, X, kickoff_windows, period_runs

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR")
                        or os.path.join(ROOT, "data"),
                        help="the match data bundle, with folders tactical/ and broadcast/ "
                             "(default: $MTAH_DATA_DIR, else ./data)")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    tracked = cleaning.load_tracking(args.data_dir)
    fused = ball.with_fused_ball(tracked, args.data_dir)
    prob, change = models.blend(tracked, fused, os.path.join(ROOT, "models"))
    labels = decode.decode(prob, tracked["period"], change)
    labels = decode.dead_before_kickoff(tracked, labels, kickoff_windows(args.data_dir))

    codes = np.full(len(labels), X, dtype=np.int8)
    for start, stop, _ in period_runs(tracked["period"]):
        codes[start:stop] = labels[start:stop]
    with open(args.out, "w", newline="") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, letter)
                          for frame, letter in enumerate(LETTERS[codes]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
