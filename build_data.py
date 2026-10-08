#!/usr/bin/env python3
"""Export one F1 session (fetched with FastF1) into static JSON for the web GUI.

Usage:
    python build_data.py --year 2026 --round 16 --session R
    python build_data.py --year 2024 --round 24 --session Q

The output is a fully static payload (data/<slug>.json) so the resulting
web app can be hosted on GitHub Pages without any server side code.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import fastf1

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------- constants
DT_MAIN = 0.5          # seconds between replay keyframes (position / progress)
DT_DETAIL = 2.0        # seconds between telemetry detail keyframes
N_TRACK = 1200         # points of the exported track polyline
N_REFS = 5             # reference laps averaged to build the track outline
REL_Q = 10_000         # quantisation of normalised lap progress (1 unit ~ 0.5 m)
XY_Q = 10              # coordinate quantisation (unit = 0.1 m)

# Session.t0_date (the wall-clock instant that SessionTime 0 refers to) is set
# by build() so absolute timestamps can be converted to session-relative seconds.
_T0: list[pd.Timestamp] = [pd.Timestamp("2000-01-01")]

STATUS_CODES = {0: "garage", 1: "on_track", 2: "off_track", 3: "stationary"}
TRACK_STATUS_NAMES = {
    "1": "green", "2": "yellow", "3": "double_yellow", "4": "safety_car",
    "5": "red", "6": "virtual_safety_car", "7": "other",
}


# ---------------------------------------------------------------- helpers
def _ms(td) -> int | None:
    """timedelta -> milliseconds (None for NaT/NaN)."""
    if td is None or (isinstance(td, float) and math.isnan(td)) or pd.isna(td):
        return None
    return int(round(td.total_seconds() * 1000))


def _sec(value) -> float:
    """timedelta **or** absolute Timestamp -> seconds relative to Session.t0_date."""
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, pd.Timestamp):
        return (value - _T0[0]).total_seconds()
    return value.total_seconds()


def _q(value, scale: int, cast=int) -> int:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return 0
    return cast(round(float(value) * scale))


def _resample_by_arc_length(xy: np.ndarray, n: int) -> np.ndarray:
    """Resample an open/closed polyline to `n` points evenly spaced in arc length."""
    seg = np.hypot(*np.diff(xy, axis=0).T)
    total = float(seg.sum())
    if total <= 0:
        return xy[:1]
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, total, n, endpoint=False)
    out = np.column_stack([
        np.interp(target, s, xy[:, 0]),
        np.interp(target, s, xy[:, 1]),
    ])
    return out


def _close_and_clean(xy: np.ndarray, min_gap: float = 0.5) -> np.ndarray:
    """Drop repeated points and close the loop."""
    keep = [xy[0]]
    for p in xy[1:]:
        if math.hypot(p[0] - keep[-1][0], p[1] - keep[-1][1]) >= min_gap:
            keep.append(p)
    if math.hypot(keep[-1][0] - keep[0][0], keep[-1][1] - keep[0][1]) < 50.0:
        keep.append(keep[0])
    return np.array(keep, dtype=float)


# ---------------------------------------------------------------- track
def build_track(session) -> dict:
    """Build the track outline from the telemetry XY of a reference lap.

    FastF1's telemetry XY coordinates live in the same coordinate space as the
    official track map data, so the outline can be taken straight from a lap.
    Extra laps are averaged in only when they start at the same physical point
    as the reference lap, because `RelativeDistance` is normalised per lap and
    a lap whose telemetry window starts elsewhere would smear the outline.
    """
    laps = session.laps
    clean = laps[(laps["Deleted"] != True) & (laps["FastF1Generated"] != True)]
    clean = clean[clean["LapTime"].notna()].sort_values("LapTime")
    if len(clean) == 0:
        raise RuntimeError("no usable lap to build the track outline")

    refs: list[np.ndarray] = []
    ref_meters = 0.0
    start_point = None
    ref_len = 0.0
    for _, lap in clean.iterrows():
        tel = lap.get_telemetry()
        if tel is None or len(tel) < 100:
            continue
        xy = tel[["X", "Y"]].to_numpy(dtype=float)
        xy = xy[~np.isnan(xy).any(axis=1)]
        if len(xy) < 100:
            continue
        arc0 = float(np.hypot(*np.diff(xy, axis=0).T).sum())
        if arc0 <= 0:
            continue
        dist = tel["Distance"].to_numpy(dtype=float)
        meters = float(np.nanmax(dist)) if not np.isnan(dist).all() else 0.0

        if start_point is None:
            # primary reference: the fastest complete lap
            start_point, ref_meters, ref_len = xy[0], meters, arc0
            refs.append(_resample_by_arc_length(xy, N_TRACK))
            continue
        if len(refs) >= N_REFS:
            break
        # only average in laps that start at the same physical point and have
        # the same length, otherwise the outline gets smeared
        if math.hypot(xy[0][0] - start_point[0], xy[0][1] - start_point[1]) > 150.0:
            continue
        if abs(arc0 - ref_len) > 0.02 * ref_len:
            continue
        refs.append(_resample_by_arc_length(xy, N_TRACK))

    if not refs:
        raise RuntimeError("no usable telemetry for the track outline")

    poly = np.mean(np.array(refs), axis=0) if len(refs) > 1 else refs[0]
    poly = _close_and_clean(poly)
    poly = _resample_by_arc_length(poly, N_TRACK)
    # a lap is a closed loop: bridge the few metres the telemetry stops short of
    poly = np.vstack([poly, poly[0]])

    xs, ys = poly[:, 0], poly[:, 1]
    seg = np.hypot(*np.diff(poly, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    arc_units = float(arc[-1])
    # calibrate the telemetry coordinate unit against the reference lap length
    # (FastF1's `Distance` column is integrated speed in meters)
    unit_meters = (ref_meters / arc_units) if arc_units > 0 and ref_meters > 0 else 0.1

    return {
        "points": [[_q(x, XY_Q), _q(y, XY_Q)] for x, y in poly],
        "bbox": {"minX": _q(xs.min(), XY_Q), "maxX": _q(xs.max(), XY_Q),
                 "minY": _q(ys.min(), XY_Q), "maxY": _q(ys.max(), XY_Q)},
        "arcLength": [round(float(v), 1) for v in arc],
        "arcUnits": round(arc_units, 1),
        "referenceLapMeters": round(ref_meters, 1),
        "unitMeters": round(unit_meters, 6),
        "referenceLaps": len(refs),
    }


def build_pitlane(pit_points: list[tuple[float, float, float]], track: dict) -> list[list[int]]:
    """Turn the collected pit-stop telemetry positions into a pit lane polyline.

    The timing system reports cars in the pit lane as `OnTrack`, so the pit
    lane is detected from the pit stop timestamps instead of from the status.
    """
    if len(pit_points) < 10:
        return []

    pts = np.array([[x * XY_Q, y * XY_Q] for _, x, y in pit_points], dtype=float)
    # order the points along the pit lane itself: project them onto the
    # principal direction of the collected pit-stop positions
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, evecs = np.linalg.eigh(centered.T @ centered)
    axis = evecs[:, -1]
    pts = pts[np.argsort(centered @ axis)]

    out: list[tuple[float, float]] = []
    for x, y in pts:
        if out and math.hypot(x - out[-1][0], y - out[-1][1]) < 20.0:
            continue
        out.append((x, y))
    if len(out) < 5:
        return []
    line = _resample_by_arc_length(np.array(out, dtype=float), 150)
    return [[int(round(x)), int(round(y))] for x, y in line]


# ---------------------------------------------------------------- drivers
def build_driver(session, abbr: str, origin: float, t_end: float) -> dict:
    """Build the time-indexed replay series for one driver."""
    laps = session.laps.pick_drivers(abbr)
    rows = []
    for _, lap in laps.iterrows():
        tel = lap.get_telemetry()
        if tel is None or len(tel) == 0:
            continue
        n = len(tel)
        lap_no = float(lap["LapNumber"]) if not pd.isna(lap["LapNumber"]) else 0.0
        t = tel["SessionTime"].dt.total_seconds().to_numpy(dtype=float) - origin
        rel = np.clip(tel["RelativeDistance"].to_numpy(dtype=float), 0.0, 1.0)
        status = tel["Status"].astype(str).to_numpy()
        st = np.where(status == "OnTrack", 1,
                     np.where(status == "Stationary", 3, 2)).astype(np.int64)
        rows.append({
            "t": t,
            "cum": (lap_no - 1.0) + rel,
            "x": tel["X"].to_numpy(dtype=float),
            "y": tel["Y"].to_numpy(dtype=float),
            "st": st,
            "speed": tel["Speed"].to_numpy(dtype=float),
            "gear": tel["nGear"].to_numpy(dtype=float),
            "drs": tel["DRS"].to_numpy(dtype=float),
            "throttle": tel["Throttle"].to_numpy(dtype=float),
            "brake": tel["Brake"].astype(bool).to_numpy(dtype=np.int64),
        })

    if not rows:
        return {}, []

    cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    order = np.argsort(cat["t"], kind="stable")
    for k in cat:
        cat[k] = cat[k][order]
    # drop duplicate timestamps so np.interp behaves
    t = cat["t"]
    uniq = np.concatenate([[True], np.diff(t) > 1e-6])
    for k in cat:
        cat[k] = cat[k][uniq]
    t = cat["t"]

    t0, t1 = float(t.min()), float(t.max())
    n_main = int(math.ceil((t_end - 0.0) / DT_MAIN)) + 1
    grid = np.arange(0, n_main) * DT_MAIN

    def nearest(values: np.ndarray, gt: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(t, gt)
        idx = np.clip(idx, 0, len(t) - 1)
        lo = np.clip(idx - 1, 0, len(t) - 1)
        pick = np.where(np.abs(t[idx] - gt) <= np.abs(t[lo] - gt), idx, lo)
        return values[pick]

    present = (grid >= t0 - 2.0) & (grid <= t1 + 2.0)
    cum = np.interp(grid, t, cat["cum"])
    lap_no = np.floor(cum) + 1
    rel = cum - np.floor(cum)
    x = np.interp(grid, t, cat["x"])
    y = np.interp(grid, t, cat["y"])
    st = nearest(cat["st"], grid)
    st = np.where(present, st, 0)

    n_detail = int(math.ceil(t_end / DT_DETAIL)) + 1
    dgrid = np.arange(0, n_detail) * DT_DETAIL

    def interp_detail(values: np.ndarray) -> np.ndarray:
        return np.interp(dgrid, t, values)

    # raw positions between a pit entry and the matching pit exit -> pit lane
    # geometry. FastF1 records PitInTime on the lap the car pits on and
    # PitOutTime on the following lap, so the two lists are paired by order.
    pit_ins = [_sec(v) - origin for v in laps["PitInTime"].dropna().sort_values()]
    pit_outs = [_sec(v) - origin for v in laps["PitOutTime"].dropna().sort_values()]
    pit_points: list[tuple[float, float, float]] = []
    for i, a in enumerate(pit_ins):
        b = pit_outs[i] if i < len(pit_outs) and pit_outs[i] > a else a + 40.0
        sel = (t >= a) & (t <= b)
        for x0, y0 in zip(cat["x"][sel], cat["y"][sel]):
            if math.isnan(x0) or math.isnan(y0):
                continue
            pit_points.append((a, float(x0), float(y0)))

    series_out = {
        "t0": round(t0, 2), "t1": round(t1, 2),
        "x": [_q(v, XY_Q) for v in x],
        "y": [_q(v, XY_Q) for v in y],
        "lap": [int(v) for v in lap_no],
        "rel": [_q(v, REL_Q) for v in rel],
        "st": [int(v) for v in st],
        "speed": [_q(v, 1) for v in interp_detail(cat["speed"])],
        "gear": [_q(v, 1) for v in interp_detail(cat["gear"])],
        "drs": [_q(v, 1) for v in interp_detail(cat["drs"])],
        "throttle": [_q(v, 1) for v in interp_detail(cat["throttle"])],
        "brake": [int(v) for v in interp_detail(cat["brake"])],
    }
    return series_out, pit_points


# ---------------------------------------------------------------- tables
def build_laps(session, origin: float) -> dict:
    out: dict[str, list[dict]] = {}
    for abbr, grp in session.laps.groupby("Driver"):
        rows = []
        for _, lap in grp.sort_values("LapNumber").iterrows():
            if pd.isna(lap["LapNumber"]):
                continue
            rows.append({
                "lap": int(lap["LapNumber"]),
                # timeline values (t0 / pitIn / pitOut) are session-relative SECONDS,
                # matching series.t0 and trackStatus.t.  Durations (time / s1..s3) stay
                # in milliseconds because the UI formats them as lap times.
                "t0": round(_sec(lap["LapStartTime"]) - origin, 2),
                "time": _ms(lap["LapTime"]),
                "s1": _ms(lap["Sector1Time"]), "s2": _ms(lap["Sector2Time"]),
                "s3": _ms(lap["Sector3Time"]),
                "pos": None if pd.isna(lap["Position"]) else int(lap["Position"]),
                "compound": None if pd.isna(lap["Compound"]) else str(lap["Compound"]),
                "tyreAge": None if pd.isna(lap["TyreLife"]) else int(lap["TyreLife"]),
                "pitIn": round(_sec(lap["PitInTime"]) - origin, 2),
                "pitOut": round(_sec(lap["PitOutTime"]) - origin, 2),
                "ts": None if pd.isna(lap["TrackStatus"]) else str(lap["TrackStatus"]),
                "del": bool(lap["Deleted"] == True),
                "gen": bool(lap["FastF1Generated"] == True),
                "acc": bool(lap["IsAccurate"] != False),
            })
        out[abbr] = rows
    return out


def build_stints(laps: dict) -> dict:
    out: dict[str, list[dict]] = {}
    for abbr, rows in laps.items():
        stints: list[dict] = []
        for r in rows:
            if not r["compound"]:
                continue
            if stints and stints[-1]["compound"] == r["compound"] and r["lap"] == stints[-1]["end"] + 1:
                stints[-1]["end"] = r["lap"]
                continue
            stints.append({"start": r["lap"], "end": r["lap"], "compound": r["compound"]})
        out[abbr] = stints
    return out


def build_track_status(session, origin: float) -> list[dict]:
    out = []
    for _, row in session.track_status.iterrows():
        t = _sec(row["Time"]) - origin
        code = str(row["Status"])
        out.append({"t": round(t, 2), "code": code,
                    "name": TRACK_STATUS_NAMES.get(code, "other"),
                    "label": str(row.get("Message", ""))})
    out.sort(key=lambda r: r["t"])
    return out


def build_race_control(session, origin: float) -> list[dict]:
    msgs = session.race_control_messages
    if msgs is None or len(msgs) == 0:
        return []
    out = []
    for _, row in msgs.iterrows():
        text = str(row.get("Message", "")).strip()
        if not text:
            continue
        flag = row.get("Flag")
        out.append({
            "t": round(_sec(row.get("Time")) - origin, 1),
            "lap": None if pd.isna(row.get("Lap")) else int(row["Lap"]),
            "flag": None if (flag is None or pd.isna(flag)) else str(flag),
            "category": None if pd.isna(row.get("Category")) else str(row["Category"]),
            "driver": None if pd.isna(row.get("RacingNumber")) else str(row["RacingNumber"]),
            "text": text,
        })
    out.sort(key=lambda r: r["t"])
    return out


# ---------------------------------------------------------------- main
def build(year: int, round_number: int, session_type: str, out_dir: str) -> str:
    t_start = time.time()
    fastf1.Cache.enable_cache(".fastf1-cache")
    session = fastf1.get_session(year, round_number, session_type)
    print(f"[1/6] loading {year} round {round_number} session {session_type} ...", flush=True)
    session.load()
    if getattr(session, "t0_date", None) is not None:
        _T0[0] = session.t0_date
    print(f"      loaded in {time.time() - t_start:.1f}s, "
          f"{len(session.drivers)} drivers, {len(session.laps)} lap rows", flush=True)

    lap_start = session.laps["LapStartTime"].dropna()
    origin = _sec(lap_start.min()) if len(lap_start) else 0.0
    last_lap = session.laps[session.laps["LapStartTime"].notna()
                            & session.laps["LapTime"].notna()].sort_values("LapStartTime")
    if len(last_lap):
        row = last_lap.iloc[-1]
        t_end = _sec(row["LapStartTime"] - pd.Timedelta(seconds=origin)) \
            + _sec(row["LapTime"]) + 45.0
    else:
        t_end = 3600.0
    print(f"      time origin (lights out) = {origin:.2f}s, replay length = {t_end:.0f}s",
          flush=True)

    print("[2/6] track outline ...", flush=True)
    track = build_track(session)
    print(f"      {len(track['points'])} points from {track['referenceLaps']} reference lap(s), "
          f"{track['arcUnits']} units / {track['referenceLapMeters']} m "
          f"-> {track['unitMeters']:.4f} m per unit", flush=True)

    print("[3/6] driver telemetry series ...", flush=True)
    series: dict[str, dict] = {}
    pit_points: list[tuple[float, float, float]] = []
    for i, abbr in enumerate(session.laps.groupby("Driver").groups.keys(), 1):
        built, pits = build_driver(session, abbr, origin, t_end)
        series[abbr] = built
        pit_points.extend(pits)
        print(f"      {i}: {abbr} ({len(built.get('x', []))} keyframes)", flush=True)

    pitlane = build_pitlane(pit_points, track)
    print(f"      pit lane points: {len(pitlane)}", flush=True)

    print("[4/6] lap tables / stints ...", flush=True)
    laps = build_laps(session, origin)
    stints = build_stints(laps)

    print("[5/6] race control / track status ...", flush=True)
    track_status = build_track_status(session, origin)
    race_control = build_race_control(session, origin)

    results = session.results
    drivers = []
    for _, r in results.sort_values("Position").iterrows():
        abbr = str(r["Abbreviation"])
        if abbr not in series:
            continue
        drivers.append({
            "abbr": abbr,
            "number": str(r["DriverNumber"]),
            "name": str(r["FullName"] or r["BroadcastName"]),
            "shortName": str(r["BroadcastName"]),
            "team": str(r["TeamName"]),
            "teamColor": str(r["TeamColor"] or "888888"),
            "grid": None if pd.isna(r["GridPosition"]) else int(r["GridPosition"]),
            "finish": None if pd.isna(r["Position"]) else int(r["Position"]),
            "status": str(r["Status"]),
            "points": None if pd.isna(r["Points"]) else float(r["Points"]),
            "laps": None if pd.isna(r["Laps"]) else int(r["Laps"]),
        })

    slug = f"{year}-r{int(round_number):02d}-{session_type.lower()}"
    payload = {
        "meta": {
            "slug": slug,
            "year": int(year),
            "round": int(round_number),
            "sessionType": session_type,
            "eventName": str(session.event["OfficialEventName"]),
            "country": str(session.event["Country"]),
            "location": str(session.event["Location"]),
            "circuit": str(session.event["Location"]),
            "date": str(session.date.date()),
            "sessionKey": str(getattr(session, "session_key", "")),
            "totalLaps": int(session.laps["LapNumber"].max() or 0),
            "duration": round(t_end, 1),
            "dtMain": DT_MAIN,
            "dtDetail": DT_DETAIL,
            "relScale": REL_Q,
            "xyScale": XY_Q,
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": f"FastF1 {fastf1.__version__}",
        },
        "track": {"outline": track["points"], "pitlane": pitlane, "bbox": track["bbox"],
                  "arcLength": track["arcLength"],
                  "arcUnits": track["arcUnits"],
                  "unitMeters": track["unitMeters"],
                  "lengthMeters": round(track["arcUnits"] * track["unitMeters"], 1),
                  "referenceLapMeters": track["referenceLapMeters"],
                  "referenceLaps": track["referenceLaps"]},
        "drivers": drivers,
        "series": series,
        "laps": laps,
        "stints": stints,
        "trackStatus": track_status,
        "raceControl": race_control,
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{slug}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(path)
    print(f"[6/6] wrote {path} ({size / 1024 / 1024:.2f} MB) in {time.time() - t_start:.1f}s",
          flush=True)
    return slug


def update_index(out_dir: str) -> None:
    entries = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json") or name == "index.json":
            continue
        path = os.path.join(out_dir, name)
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            meta = data["meta"]
        except Exception as exc:
            print(f"index: skipping {name}: {exc}")
            continue
        entries.append({
            "slug": meta["slug"], "file": name,
            "label": f"{meta['year']} {meta['country']} ({meta['sessionType']})",
            "eventName": meta["eventName"], "date": meta["date"],
            "drivers": len(meta.get("drivers", [])) or len(data.get("drivers", [])),
            "sizeKB": round(os.path.getsize(os.path.join(out_dir, name)) / 1024),
        })
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"sessions": entries}, fh, ensure_ascii=False, indent=1)
    print(f"index: {len(entries)} session(s) -> {os.path.join(out_dir, 'index.json')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="FastF1 -> static JSON export")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--session", default="R", help="R / Q / S / SQ / FP1 ...")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "data"))
    ap.add_argument("--no-index", action="store_true")
    args = ap.parse_args()
    build(args.year, args.round, args.session.upper(), args.out)
    if not args.no_index:
        update_index(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
