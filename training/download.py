"""Download the two public datasets the models are trained on (other matches only), about 4.5 GB.

    python -m training.download

dfl      Bassek et al., "An integrated dataset of spatiotemporal and event data in elite soccer",
         Scientific Data 2025. 7 Bundesliga / 2. Bundesliga matches, 25 fps tracking with a
         per-frame ball-status and ball-possession flag. CC BY 4.0.
         https://doi.org/10.6084/m9.figshare.28196177
metrica  Metrica Sports sample data, 3 anonymised matches, 25 fps tracking and events.
         https://github.com/metrica-sports/sample-data

Every file is pinned (a commit id or a figshare file id) and checked against the SHA-256 in
training/external_checksums.json.
"""

import hashlib
import json
import os
import urllib.request

from .common import EXTERNAL_DIR

CHECKSUMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_checksums.json")

METRICA_SHA = "e706dd506b360d69d9d123d5b8026e7294b13996"
METRICA_RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/" + METRICA_SHA + "/data/"
METRICA_FILES = (
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "Sample_Game_3/Sample_Game_3_events.json",
    "Sample_Game_3/Sample_Game_3_metadata.xml",
    "Sample_Game_3/Sample_Game_3_tracking.txt",
)

# figshare article 28196177, version 1: (competition, match, matchinfo id, events id, positions id)
DFL_FIGSHARE = "https://ndownloader.figshare.com/files/"
DFL_MATCHES = (
    ("DFL-COM-000001", "DFL-MAT-J03WMX", 51643475, 51643493, 51643514),
    ("DFL-COM-000001", "DFL-MAT-J03WN1", 51643472, 51643496, 51643517),
    ("DFL-COM-000002", "DFL-MAT-J03WOH", 51643478, 51643499, 51643520),
    ("DFL-COM-000002", "DFL-MAT-J03WOY", 51643481, 51643502, 51643523),
    ("DFL-COM-000002", "DFL-MAT-J03WPY", 51643487, 51643505, 51643526),
    ("DFL-COM-000002", "DFL-MAT-J03WQQ", 51643484, 51643508, 51643529),
    ("DFL-COM-000002", "DFL-MAT-J03WR9", 51643490, 51643511, 51643532),
)


def file_list():
    """[(relative path under EXTERNAL_DIR, url)]."""
    out = [("metrica/" + name, METRICA_RAW + name) for name in METRICA_FILES]
    for competition, match, info_id, events_id, positions_id in DFL_MATCHES:
        tail = "_{}_{}.xml".format(competition, match)
        out.append(("dfl/DFL_02_01_matchinformation" + tail, DFL_FIGSHARE + str(info_id)))
        out.append(("dfl/DFL_03_02_events_raw" + tail, DFL_FIGSHARE + str(events_id)))
        out.append(("dfl/DFL_04_03_positions_raw_observed" + tail, DFL_FIGSHARE + str(positions_id)))
    return out


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = path + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    os.replace(partial, path)


def main():
    with open(CHECKSUMS) as handle:
        checksums = json.load(handle)
    for relative, url in file_list():
        path = os.path.join(EXTERNAL_DIR, relative)
        if not os.path.isfile(path):
            print("fetching {}".format(relative), flush=True)
            for attempt in range(3):
                try:
                    fetch(url, path)
                    break
                except Exception as error:  # network errors: retry, then give up loudly
                    print("  attempt {} failed: {}".format(attempt + 1, error), flush=True)
            else:
                raise SystemExit("could not download {}".format(url))
        if sha256_of(path) != checksums[relative]:
            raise SystemExit("checksum mismatch for {}; delete the file and run again".format(
                relative))
    print("external data ready in {}".format(EXTERNAL_DIR))


if __name__ == "__main__":
    main()
