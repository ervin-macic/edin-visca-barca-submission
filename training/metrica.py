"""Metrica Sports sample data (3 matches) -> the common match format.

Tracking is Metrica's corrected data at 25 fps in the same normalised 0-1 coordinates as the
competition ATD. Labels are derived mechanically from the event files (definitions:
documentation/events-definitions.pdf in the Metrica repository):

  team   the team of the most recent on-ball event (PASS, CARRY, SHOT, RECOVERY, SET PIECE,
         BALL LOST, BALL OUT). A turnover therefore happens on the frame of the RECOVERY event,
         i.e. when the other team first controls the ball. CHALLENGE events (duels) do not move
         possession by themselves. A BALL LOST / BALL OUT logged within a second after the other
         team's own on-ball event is the loser's side of that same turnover and is ignored.
  dead   from the frame after a stopping event ends up to the frame before the next restart.
         Stops: BALL OUT; a shot that goes out or scores; a foul (FAULT RECEIVED, or a CHALLENGE
         with FAULT); offside, handball, FORCED (play halted by the referee) and END HALF on any
         on-ball event. Restarts: after the ball left the pitch, the next SET PIECE, PASS, CARRY,
         SHOT, BALL LOST or BALL OUT; after a whistle (foul, offside, handball, forced stop, end
         of half) only a SET PIECE or a GOAL KICK, because events logged in the seconds after a
         whistle are not a restart.
         After a whistle the dead ball starts where these files stop giving a ball position (they
         carry no position while the ball is dead), if that happens within 4 s of the event.

In these files the ball has no position during most dead time. Those stretches are filled with a
straight line between the last and next position, so that "no ball position" is not a give-away of
dead ball; ball losses are then put back in by augment.py under explicit control.

    python -m training.metrica       # converts the three games into cache/ext/metrica_*.npz
"""

import csv
import json
import os
import re

import numpy as np

from .common import A, D, EXTERNAL_DIR, H, X, cache_path, label_summary, save_match

PITCH = (105.0, 68.0)
BASE = os.path.join(EXTERNAL_DIR, "metrica")

ON_BALL = ("PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "BALL OUT")
CONTROL = ("PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE")
# Actions that prove the ball is in play again. RECOVERY is left out on purpose: after the ball
# goes out, a goalkeeper or ball boy picking it up is logged as a RECOVERY while play is stopped.
# BALL OUT is in the list because the event starts with a kick (for example a goal kick that goes
# straight out): the ball is alive from its start frame and dead again after its end frame.
RESTART = ("SET PIECE", "PASS", "CARRY", "SHOT", "BALL LOST", "BALL OUT")
WHISTLE_WORDS = ("FAULT", "FORCED", "OFFSIDE", "HAND BALL", "END HALF")
ECHO_FRAMES = 25            # a loser's BALL LOST this soon after the winner's event is an echo
WHISTLE_SEARCH = 100        # frames after a whistle event in which the ball position may end


def stops_play(kind, subtype):
    """True when an event ends with the ball dead."""
    if kind == "BALL OUT" or kind == "FAULT RECEIVED":
        return True
    if kind == "SHOT" and ("OUT" in subtype or "GOAL" in subtype):
        return True
    if kind == "CHALLENGE":
        return "FAULT" in subtype
    if kind in ("BALL LOST", "PASS", "CARRY", "SHOT"):
        return any(word in subtype for word in ("OFFSIDE", "HAND BALL", "END HALF", "FORCED"))
    return False


def is_whistle(kind, subtype):
    """True for stops made by the referee rather than by the ball leaving the pitch."""
    return kind == "FAULT RECEIVED" or any(word in subtype for word in WHISTLE_WORDS)


def to_metres(xy):
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = (xy[..., 0] - 0.5) * PITCH[0]
    out[..., 1] = (xy[..., 1] - 0.5) * PITCH[1]
    return out


def read_csv_tracking(path):
    """(period, players (T, P, 2), ball (T, 2)) from one team's tracking CSV, normalised units."""
    with open(path) as handle:
        rows = list(csv.reader(handle))[3:]
    table = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in row] for row in rows],
                     dtype=np.float64)
    period = table[:, 0].astype(np.int8)
    xy = table[:, 3:]
    players = xy[:, :-2].reshape(len(xy), -1, 2)
    return period, players, xy[:, -2:]


def read_csv_events(path):
    """[(start_frame, end_frame, team 'H'/'A', type, subtype)] with 0-based frames."""
    out = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            start = int(row["Start Frame"]) - 1
            end = max(int(row["End Frame"]) - 1, start)
            out.append((start, end, "H" if row["Team"] == "Home" else "A",
                        row["Type"], row["Subtype"]))
    return out


def labels_from_events(events, period, ball_missing=None):
    """Per-frame H / A / D from the event list, following the rules in the module docstring."""
    n = len(period)
    label = np.full(n, X, dtype=np.int8)
    events = sorted(events, key=lambda e: (e[0], e[1]))

    owner = np.full(n, -1, dtype=np.int8)       # team of the latest on-ball event
    last_control = {H: -10 ** 9, A: -10 ** 9}
    for start, _, team, kind, subtype in events:
        if kind not in ON_BALL or not 0 <= start < n:
            continue
        code = H if team == "H" else A
        other = A if code == H else H
        if (kind in ("BALL LOST", "BALL OUT") and "GOAL KICK" not in subtype
                and start - last_control[other] <= ECHO_FRAMES):
            continue
        owner[start] = code
        if kind in CONTROL:
            last_control[code] = start
    index = np.where(owner >= 0, np.arange(n), -1)
    np.maximum.accumulate(index, out=index)
    filled = np.where(index >= 0, owner[np.maximum(index, 0)], -1)

    dead = np.zeros(n, dtype=bool)
    dead_from, whistle = None, False
    for start, end, _, kind, subtype in events:
        if dead_from is not None and kind in RESTART and start >= dead_from:
            if not whistle or kind == "SET PIECE" or "GOAL KICK" in subtype:
                dead[dead_from:start] = True
                dead_from = None
        if stops_play(kind, subtype) and dead_from is None:
            dead_from, whistle = end + 1, is_whistle(kind, subtype)
            if whistle and ball_missing is not None:
                ahead = np.flatnonzero(ball_missing[dead_from:dead_from + WHISTLE_SEARCH])
                if len(ahead):
                    dead_from += int(ahead[0])
    if dead_from is not None:
        dead[dead_from:] = True

    # Frames before the first event of a half (and after a half's last whistle) are dead.
    for half in np.unique(period[period > 0]):
        frames = np.flatnonzero(period == half)
        first, last = frames[0], frames[-1]
        starts = [s for s, _, _, kind, _ in events if kind in RESTART and first <= s <= last]
        if starts:
            dead[first:min(starts)] = True
        dead[first:last + 1] |= filled[first:last + 1] < 0

    in_half = period > 0
    label[in_half] = np.where(dead[in_half], D, filled[in_half])
    label[in_half & (label == -1)] = D
    return label


def fill_dead_ball(ball, label):
    """Ball track with the missing positions inside dead-ball runs bridged by a straight line."""
    out = ball.copy()
    seen = np.isfinite(ball[:, 0])
    index = np.flatnonzero(seen)
    for a, b in zip(index[:-1], index[1:]):
        if b - a > 1 and np.any(label[a + 1:b] == D):
            t = ((np.arange(a + 1, b) - a) / float(b - a))[:, None]
            out[a + 1:b] = ball[a] + t * (ball[b] - ball[a])
    return out


def finish(ball, home, away, events, period):
    ball = to_metres(ball)
    label = labels_from_events(events, period, ~np.isfinite(ball[:, 0]))
    return {"ball": fill_dead_ball(ball, label), "home": to_metres(home), "away": to_metres(away),
            "label": label, "period": period, "pitch": np.array(PITCH, dtype=np.float32),
            "ball_as_published": ball}


def load_csv_game(number):
    folder = os.path.join(BASE, "Sample_Game_{}".format(number))
    stem = os.path.join(folder, "Sample_Game_{}_".format(number))
    period, home, ball = read_csv_tracking(stem + "RawTrackingData_Home_Team.csv")
    _, away, _ = read_csv_tracking(stem + "RawTrackingData_Away_Team.csv")
    events = read_csv_events(stem + "RawEventsData.csv")
    return finish(ball, home, away, events, period)


def load_epts_game(number=3):
    folder = os.path.join(BASE, "Sample_Game_{}".format(number))
    stem = os.path.join(folder, "Sample_Game_{}_".format(number))
    with open(stem + "metadata.xml", encoding="utf-8") as handle:
        meta = handle.read()

    team_of_player = dict(re.findall(r'<Player id="([^"]+)" teamId="([^"]+)"', meta))
    team_ids = re.findall(r'<Team id="([^"]+)"', meta)
    home_id = team_ids[0]

    # The raw file changes its column layout whenever the players on the pitch change; each
    # DataFormatSpecification block lists the player channels valid for a frame range.
    specs = []
    for first, last, body in re.findall(
            r'<DataFormatSpecification endFrame="(\d+)"[^>]*startFrame="(\d+)">(.*?)'
            r'</DataFormatSpecification>', meta, flags=re.S):
        channels = re.findall(r'<PlayerChannelRef playerChannelId="([^"]+)"', body)
        players = [c[:-2] for c in channels[::2]]
        specs.append((int(last), int(first), players))
    channel_player = dict(re.findall(r'<PlayerChannel channelId="x" id="([^"]+)_x" playerId="([^"]+)"',
                                     meta))

    all_players = sorted(team_of_player)
    home_players = [p for p in all_players if team_of_player[p] == home_id]
    away_players = [p for p in all_players if team_of_player[p] != home_id]
    slot = {p: i for i, p in enumerate(home_players)}
    slot.update({p: i for i, p in enumerate(away_players)})

    with open(stem + "tracking.txt") as handle:
        lines = handle.read().split("\n")
    lines = [line for line in lines if line]
    n = int(lines[-1].split(":")[0])
    home = np.full((n, len(home_players), 2), np.nan)
    away = np.full((n, len(away_players), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for line in lines:
        frame_text, players_text, ball_text = line.split(":")
        frame = int(frame_text)
        names = None
        for first, last, players in specs:
            if first <= frame <= last:
                names = players
                break
        for name, group in zip(names, players_text.split(";")):
            x, y = group.split(",")[:2]
            player = channel_player[name]
            target = home if team_of_player[player] == home_id else away
            target[frame - 1, slot[player]] = (float(x), float(y))
        bx, by = ball_text.split(",")[:2]
        ball[frame - 1] = (float(bx), float(by))

    with open(stem + "events.json") as handle:
        raw = json.load(handle)["data"]
    events = []
    half_start = {}
    half_end = {}
    for event in raw:
        subtypes = event.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        subtype = "-".join(s["name"] for s in subtypes)
        start = int(event["start"]["frame"]) - 1
        end = max(int(event["end"]["frame"]) - 1, start)
        team = "H" if event["team"]["id"] == home_id else "A"
        events.append((start, end, team, event["type"]["name"], subtype))
        half = int(event["period"])
        half_start[half] = min(half_start.get(half, start), start)
        half_end[half] = max(half_end.get(half, end), end)

    period = np.zeros(n, dtype=np.int8)
    for half in half_start:
        period[half_start[half]:min(half_end[half], n - 1) + 1] = half

    return finish(ball, home, away, events, period)


def load(number):
    return load_epts_game(number) if number == 3 else load_csv_game(number)


if __name__ == "__main__":
    for number in (1, 2, 3):
        match = load(number)
        save_match(cache_path("ext", "metrica_{}.npz".format(number)), match)
        summary = label_summary(match)
        nan = np.isnan(match["ball_as_published"][:, 0])
        graded = match["period"] > 0
        print("metrica", number, {k: round(v, 3) for k, v in summary.items()},
              "| published ball missing: alive {:.3f} dead {:.3f}".format(
                  float(nan[graded & (match["label"] != D)].mean()),
                  float(nan[graded & (match["label"] == D)].mean())), flush=True)
