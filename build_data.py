#!/usr/bin/env python3
"""Export F1 sessions (fetched with FastF1) into static JSON for the web GUI.

Single session:
    python build_data.py --year 2026 --round 16 --session R

Whole history (2018 -> latest, Race / Sprint / Sprint Qualifying / Qualifying):
    python build_data.py --bulk --workers 6
    python build_data.py --bulk --years 2018 2019 --types R Q

Outputs:
    data/<slug>.json.gz      full resolution replay payload (gzip)
    .features/<slug>.json    per session machine learning feature tables
    data/index.json          session catalog used by the GUI

The output is fully static so the resulting web app can be hosted on GitHub
Pages without any server side code.  `build_features.py` merges the per session
feature tables into CSV / Parquet tables.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
import traceback
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
PIT_MIN_M = 60.0      # metres; below this the timing data is not a real pit lane
PIT_MAX_M = 1200.0    # metres; above this the segment ran onto the circuit

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


def _sec_rel(value, origin: float) -> float | None:
    """Same as `_sec(value) - origin` but returns **None** for NaT/NaN.

    `build_pit_stops` and the feature tables rely on `is None` guards, so a bare
    NaN here silently turns every lap into a pit stop.
    """
    if value is None or pd.isna(value):
        return None
    v = _sec(value) - origin
    return round(v, 2) if math.isfinite(v) else None


def _q(value, scale: int, cast=int) -> int:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return 0
    return cast(round(float(value) * scale))


def _num(value) -> float:
    """NaN-safe float conversion."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _lap_telemetry(lap):
    """Return a telemetry frame for one lap, or None when nothing is available.

    `Lap.get_telemetry()` merges the car data with the positional data.  Some
    early 2018 sessions have no positional data at all, which makes
    `get_telemetry()` raise KeyError.  In that case the car data is used and a
    synthetic per-lap progress is derived from the integrated speed, so the lap
    tables, the stint tables and the telemetry features are still exported.
    Such laps have no X/Y and are flagged by `meta.hasPosition = false`.
    """
    tel = None
    try:
        tel = lap.get_telemetry()
    except Exception:
        tel = None
    if tel is not None and len(tel) and "X" in tel.columns:
        return tel

    try:
        car = lap.get_car_data()
    except Exception:
        return None
    if car is None or len(car) == 0 or "Speed" not in car.columns:
        return None

    car = car.copy()
    if "Distance" not in car.columns:
        try:
            # `add_distance()` returns a new joined frame instead of mutating,
            # so the result has to be captured explicitly.
            joined = car.add_distance()
            if joined is not None and "Distance" in joined.columns:
                car = joined
            else:
                raise KeyError("Distance")
        except Exception:
            t = car["SessionTime"].dt.total_seconds().to_numpy(dtype=float)
            v = np.nan_to_num(car["Speed"].to_numpy(dtype=float), nan=0.0)
            dt = np.abs(np.diff(t, prepend=t[0]))
            car["Distance"] = np.cumsum(dt * v)
    dist = car["Distance"].to_numpy(dtype=float)
    top = float(np.nanmax(dist)) if np.isfinite(dist).any() else 0.0
    if "RelativeDistance" not in car.columns:
        car["RelativeDistance"] = (dist / top) if top > 0 else 0.0
    car["X"] = np.nan
    car["Y"] = np.nan
    if "Status" not in car.columns:
        car["Status"] = "OnTrack"
    return car


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
        return None

    refs: list[np.ndarray] = []
    ref_meters = 0.0
    start_point = None
    ref_len = 0.0
    for _, lap in clean.iterrows():
        tel = _lap_telemetry(lap)
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
        return None

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
    if not pit_points:
        return []

    # group the raw samples by pit stop (the key is that stop's pit-in time);
    # inside one stop the samples are already in time order, which is the order
    # the car actually drove through the pit lane
    groups: dict[float, list[tuple[float, float]]] = {}
    for key, x, y in pit_points:
        if math.isnan(x) or math.isnan(y):
            continue
        groups.setdefault(key, []).append((x * XY_Q, y * XY_Q))
    # the stop with the most samples gives the most complete path
    candidates = sorted((g for g in groups.values() if len(g) >= 8), key=len, reverse=True)
    if not candidates:
        return []

    def clean(points: np.ndarray) -> np.ndarray:
        out: list[tuple[float, float]] = []
        for x, y in points:
            # 2 m in exported units (1/100 m) collapses the standing still samples
            if out and math.hypot(x - out[-1][0], y - out[-1][1]) < 2.0 * XY_Q * XY_Q:
                continue
            out.append((x, y))
        return np.array(out, dtype=float)

    def length(points: np.ndarray) -> float:
        """Polyline length in metres; the points are in exported 1/100 m units."""
        return float(np.hypot(*np.diff(points, axis=0).T).sum()) / (XY_Q * XY_Q)

    cleaned = [clean(np.array(group, dtype=float)) for group in candidates]
    valid = []
    for i, line in enumerate(cleaned):
        if len(line) < 5:
            continue
        # A real pit lane traverse is a few hundred metres.  Too short means the
        # timing system stopped tracking the car inside the pit lane (common in
        # qualifying), too long means the pit-out pairing ran onto the circuit.
        # In both cases no pit lane is exported rather than a misleading stub.
        dist = length(line)
        if not (PIT_MIN_M <= dist <= PIT_MAX_M):
            continue
        valid.append((dist, i, line))
    if not valid:
        return []
    # the longest valid traverse shows the whole lane, not just one segment
    valid.sort(key=lambda v: -v[0])
    base_len, base_index, base = valid[0]

    # average with further stops that trace the same lane, so lane changes and
    # braking noise cancel out.  Candidates are compared point by point after
    # resampling, which is what actually keeps the shape honest.
    base_line = _resample_by_arc_length(base, 150)
    lines = [base_line]
    for i, line in enumerate(cleaned):
        if i == base_index or len(lines) >= 5 or len(line) < 5:
            continue
        if abs(length(line) - base_len) > 0.25 * base_len:
            continue
        cand = _resample_by_arc_length(line, 150)
        if float(np.mean(np.hypot(*(cand - base_line).T))) > 15.0:
            continue
        lines.append(cand)

    return [[int(round(x)), int(round(y))] for x, y in np.mean(np.stack(lines), axis=0)]


# ---------------------------------------------------------------- drivers
def _col(tel, name: str, default: float = 0.0, as_bool: bool = False) -> np.ndarray:
    """Read one telemetry column as a float array, tolerating missing columns."""
    if name not in tel.columns:
        return np.full(len(tel), default, dtype=float)
    if as_bool:
        return pd.to_numeric(tel[name].astype(bool), errors="coerce").fillna(0).to_numpy(dtype=float)
    return pd.to_numeric(tel[name], errors="coerce").to_numpy(dtype=float)


def _sample_durations(t: np.ndarray) -> np.ndarray:
    """Seconds each telemetry sample is assumed to hold (used for time-weighted stats)."""
    n = len(t)
    dt = np.zeros(n, dtype=float)
    if n >= 2:
        dt[1:] = np.diff(t)
        dt[0] = dt[1]
    return np.clip(dt, 0.0, 5.0)


def _series_stats(cat: dict, t: np.ndarray) -> dict:
    """Machine learning features computed from the raw (un-resampled) telemetry."""
    n = len(t)
    if n == 0:
        return {}
    dt = _sample_durations(t)
    total = float(dt.sum())

    def wstats(values: np.ndarray) -> tuple[float, float, float, float]:
        """(mean, p95, max, min) over the samples, NaN ignored."""
        v = values[np.isfinite(values)]
        if v.size == 0:
            return (float("nan"),) * 4
        return (float(np.mean(v)), float(np.percentile(v, 95)),
                float(np.max(v)), float(np.min(v)))

    speed = np.nan_to_num(cat["speed"], nan=0.0)
    throttle = np.nan_to_num(cat["throttle"], nan=0.0)
    brake = cat["brake"] > 0.5
    drs = np.nan_to_num(cat["drs"], nan=0.0)
    gear = np.nan_to_num(cat["gear"], nan=0.0)
    st = cat["st"]

    s_mean, s_p95, s_max, s_min = wstats(cat["speed"])
    th_mean, th_p95, th_max, _ = wstats(cat["throttle"])

    # acceleration in m/s^2 from the speed trace (speed is km/h)
    v = speed / 3.6
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.diff(v) / np.where(dt[1:] > 0, dt[1:], np.nan)
    acc = acc[np.isfinite(acc)]

    gear_changes = int(np.count_nonzero(np.diff(gear) != 0))
    drs_zone = float(dt[drs >= 1].sum())
    drs_open = float(dt[drs >= 2].sum())

    stats = {
        "telemetrySamples": n,
        "telemetrySeconds": round(total, 2),
        "speedMaxKmh": round(s_max, 2),
        "speedMinKmh": round(s_min, 2),
        "speedMeanKmh": round(s_mean, 2),
        "speedP95Kmh": round(s_p95, 2),
        "speedOver300Seconds": round(float(dt[speed >= 300].sum()), 2),
        "speedOver250Seconds": round(float(dt[speed >= 250].sum()), 2),
        "speedOver200Seconds": round(float(dt[speed >= 200].sum()), 2),
        "speedOver100Seconds": round(float(dt[speed >= 100].sum()), 2),
        "throttleMeanPct": round(th_mean, 2),
        "throttleP95Pct": round(th_p95, 2),
        "throttleMaxPct": round(th_max, 2),
        "throttleOver50Seconds": round(float(dt[throttle >= 50].sum()), 2),
        "throttleZeroSeconds": round(float(dt[throttle <= 1].sum()), 2),
        "brakeSeconds": round(float(dt[brake].sum()), 2),
        "brakeRatioPct": round(100.0 * float(dt[brake].sum()) / total, 2) if total > 0 else 0.0,
        "drsZoneSeconds": round(drs_zone, 2),
        "drsOpenSeconds": round(drs_open, 2),
        "drsOpenRatioPct": round(100.0 * drs_open / total, 2) if total > 0 else 0.0,
        "gearMax": int(gear.max()),
        "gearMin": int(gear.min()),
        "gearChanges": gear_changes,
        "accelMaxMs2": round(float(np.max(acc)), 2) if acc.size else 0.0,
        "accelP95Ms2": round(float(np.percentile(acc, 95)), 2) if acc.size else 0.0,
        "decelMaxMs2": round(float(np.min(acc)), 2) if acc.size else 0.0,
        "decelP05Ms2": round(float(np.percentile(acc, 5)), 2) if acc.size else 0.0,
        "stationarySeconds": round(float(dt[st == 3].sum()), 2),
        "offTrackSeconds": round(float(dt[st == 2].sum()), 2),
        "onTrackSeconds": round(float(dt[st == 1].sum()), 2),
    }
    return stats


# ---------------------------------------------------------------- drivers
def build_driver(session, abbr: str, origin: float, t_end: float) -> tuple[dict, list, dict]:
    """Build the time-indexed replay series for one driver.

    Returns (series, pit-lane points, raw telemetry feature stats).
    """
    laps = session.laps.pick_drivers(abbr)
    rows = []
    for _, lap in laps.iterrows():
        tel = _lap_telemetry(lap)
        if tel is None or len(tel) == 0:
            continue
        lap_no = float(lap["LapNumber"]) if not pd.isna(lap["LapNumber"]) else 0.0
        t = tel["SessionTime"].dt.total_seconds().to_numpy(dtype=float) - origin
        rel = np.nan_to_num(np.clip(_col(tel, "RelativeDistance"), 0.0, 1.0), nan=0.0)
        status = tel["Status"].astype(str).to_numpy() if "Status" in tel.columns else np.full(len(tel), "OnTrack")
        st = np.where(status == "OnTrack", 1,
                     np.where(status == "Stationary", 3, 2)).astype(np.int64)
        rows.append({
            "t": t,
            "cum": (lap_no - 1.0) + rel,
            "x": _col(tel, "X", default=float("nan")),
            "y": _col(tel, "Y", default=float("nan")),
            "st": st,
            "speed": _col(tel, "Speed", default=float("nan")),
            "gear": _col(tel, "nGear", default=0.0),
            "drs": _col(tel, "DRS", default=0.0),
            "throttle": _col(tel, "Throttle", default=0.0),
            "brake": _col(tel, "Brake", default=0.0, as_bool=True),
        })

    if not rows:
        return {}, [], {}

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

    stats = _series_stats(cat, t)
    has_pos = bool(np.isfinite(cat["x"]).any() and np.isfinite(cat["y"]).any())

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
    # `cum` is the continuous lap progress used for the ranking; a single NaN
    # here would turn into `int(NaN)` below and abort the whole session.
    cum = np.nan_to_num(np.interp(grid, t, cat["cum"]), nan=0.0)
    lap_no = np.nan_to_num(np.floor(cum) + 1, nan=0.0)
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
        # without positional data the coordinates stay 0 and `hasPos` is false,
        # so the GUI skips the map and shows the analysis views instead
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
        "hasPos": has_pos,
    }
    stats["hasPosition"] = int(has_pos)
    return series_out, pit_points, stats


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
                "t0": _sec_rel(lap["LapStartTime"], origin),
                "time": _ms(lap["LapTime"]),
                "s1": _ms(lap["Sector1Time"]), "s2": _ms(lap["Sector2Time"]),
                "s3": _ms(lap["Sector3Time"]),
                "pos": _int_or(lap.get("Position")),
                "compound": None if pd.isna(lap["Compound"]) else str(lap["Compound"]),
                "tyreAge": _int_or(lap.get("TyreLife")),
                "pitIn": _sec_rel(lap["PitInTime"], origin),
                "pitOut": _sec_rel(lap["PitOutTime"], origin),
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
        t = _sec_rel(row["Time"], origin)
        if t is None:
            continue
        code = str(row["Status"])
        out.append({"t": t, "code": code,
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
        t_rel = _sec_rel(row.get("Time"), origin)
        if t_rel is None:
            continue
        out.append({
            "t": t_rel,
            "lap": _int_or(row.get("Lap")),
            "flag": None if (flag is None or pd.isna(flag)) else str(flag),
            "category": None if pd.isna(row.get("Category")) else str(row["Category"]),
            "driver": None if pd.isna(row.get("RacingNumber")) else str(row["RacingNumber"]),
            "text": text,
        })
    out.sort(key=lambda r: r["t"])
    return out


# ---------------------------------------------------------------- feature tables
import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed

SESSION_NAME_TO_CODE = {
    "Race": "R", "Feature Race": "R", "Main Race": "R",
    "Sprint": "S", "Sprint Qualifying": "SQ", "Qualifying": "Q",
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
}
SESSION_ORDER = {"SQ": 0, "Q": 1, "S": 2, "R": 3}
SESSION_LABELS = {"R": "決勝", "Q": "予選", "S": "スプリント", "SQ": "スプリント予選"}

FEATURE_TABLES = ["sessions", "session_results", "laps", "stints", "pit_stops",
                  "driver_session_features", "telemetry_stats", "track_status",
                  "race_control", "weather"]


def build_weather(session, origin: float) -> list[dict]:
    """session.weather_data -> time-indexed weather rows (ML features)."""
    data = getattr(session, "weather_data", None)
    if data is None or len(data) == 0:
        return []
    out = []
    for _, row in data.iterrows():
        t = _sec_rel(row["Time"], 0.0)
        if t is None:
            continue
        out.append({
            "t": t,
            "rel": round(t - origin, 1),
            "airTemp": _num(row.get("AirTemp")),
            "trackTemp": _num(row.get("TrackTemp")),
            "humidity": _num(row.get("Humidity")),
            "pressure": _num(row.get("Pressure")),
            "rainfall": _num(row.get("Rainfall")),
            "windSpeed": _num(row.get("WindSpeed")),
            "windDirection": _num(row.get("WindDirection")),
        })
    out.sort(key=lambda r: r["t"])
    return out


def _weather_at(rows: list[dict], t: float) -> dict:
    if not rows:
        return {}
    ts = [r["t"] for r in rows]
    i = bisect.bisect_right(ts, t) - 1
    return rows[i] if i >= 0 else {}


def build_pit_stops(laps: dict) -> dict:
    """Derive one row per pit stop from the lap table (PitInTime / PitOutTime)."""
    out: dict[str, list[dict]] = {}
    for abbr, rows in laps.items():
        by_lap = {r["lap"]: r for r in rows}
        stops: list[dict] = []
        for r in rows:
            if r["pitIn"] is None:
                continue
            pit_out = None
            nxt = by_lap.get(r["lap"] + 1)
            if nxt is not None:
                pit_out = nxt["pitOut"]
            if pit_out is None:
                for cand in rows:
                    if cand["lap"] > r["lap"] and cand["pitOut"] is not None:
                        pit_out = cand["pitOut"]
                        break
            dur = None if pit_out is None else round(pit_out - r["pitIn"], 2)
            stops.append({
                "stop": len(stops) + 1,
                "lapBefore": r["lap"],
                "pitIn": r["pitIn"],
                "pitOut": pit_out,
                "stopSeconds": dur,
                "lapTimeBeforeMs": r["time"],
                "compoundBefore": r["compound"],
                "tyreAgeBefore": r["tyreAge"],
            })
        out[abbr] = stops
    return out


def _aggregate_track_status(track_status: list[dict]) -> tuple[dict, dict]:
    counts: dict[str, int] = {}
    seconds: dict[str, float] = {}
    for i, row in enumerate(track_status):
        counts[row["name"]] = counts.get(row["name"], 0) + 1
        if i + 1 < len(track_status):
            dt = max(0.0, track_status[i + 1]["t"] - row["t"])
            seconds[row["name"]] = seconds.get(row["name"], 0.0) + dt
    return counts, seconds


def _weather_summary(weather: list[dict]) -> dict:
    if not weather:
        return {}

    def stat(key: str) -> tuple[float | None, float | None, float | None]:
        vals = [r[key] for r in weather if math.isfinite(r[key])]
        if not vals:
            return None, None, None
        return (round(sum(vals) / len(vals), 2), round(min(vals), 2), round(max(vals), 2))

    rain = [r["rainfall"] for r in weather if math.isfinite(r["rainfall"])]
    rain_total = round(max(rain), 2) if rain else None
    mean_air, min_air, max_air = stat("airTemp")
    mean_track, min_track, max_track = stat("trackTemp")
    mean_hum, min_hum, max_hum = stat("humidity")
    mean_press, min_press, max_press = stat("pressure")
    mean_wind, min_wind, max_wind = stat("windSpeed")
    return {
        "weatherSamples": len(weather),
        "avgAirTempC": mean_air, "minAirTempC": min_air, "maxAirTempC": max_air,
        "avgTrackTempC": mean_track, "minTrackTempC": min_track, "maxTrackTempC": max_track,
        "avgHumidityPct": mean_hum, "minHumidityPct": min_hum, "maxHumidityPct": max_hum,
        "avgPressureHpa": mean_press, "minPressureHpa": min_press, "maxPressureHpa": max_press,
        "avgWindSpeedMs": mean_wind, "minWindSpeedMs": min_wind, "maxWindSpeedMs": max_wind,
        "rainfallTotalMm": rain_total,
        "rainfallSamples": sum(1 for v in rain if v > 0),
        "isDry": bool(rain_total is not None and rain_total <= 0.0),
    }


def _driver_aggregates(rows: list[dict], stops: list[dict], driver_stints: list[dict]) -> dict:
    valid = [r for r in rows if r["time"] is not None and not r["del"]]
    clean = [r for r in valid if not r["gen"]]
    out: dict = {
        "lapRows": len(rows),
        "lapsCompleted": len(valid),
        "pitStops": len(stops),
        "stintCount": len(driver_stints),
    }
    if valid:
        times = [r["time"] for r in valid]
        out["bestLapMs"] = min(times)
        out["avgLapMs"] = round(sum(times) / len(times), 1)
        out["stddevLapMs"] = round((sum((t - out["avgLapMs"]) ** 2 for t in times) / len(times)) ** 0.5, 1)
        out["lastLapMs"] = valid[-1]["time"]
    if clean:
        out["bestLapValidMs"] = min(r["time"] for r in clean)
    deleted = [r for r in rows if r["del"]]
    out["deletedLaps"] = len(deleted)
    if deleted:
        out["deletedLapMsSaved"] = sum(r["time"] for r in deleted if r["time"] is not None)
    sectors = {
        "bestS1Ms": min((r["s1"] for r in valid if r["s1"] is not None), default=None),
        "bestS2Ms": min((r["s2"] for r in valid if r["s2"] is not None), default=None),
        "bestS3Ms": min((r["s3"] for r in valid if r["s3"] is not None), default=None),
    }
    out.update(sectors)
    compounds = [r["compound"] for r in rows if r["compound"]]
    counts: dict[str, int] = {}
    for c in compounds:
        counts[c] = counts.get(c, 0) + 1
    out["compoundsUsed"] = ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    out["compoundStintCount"] = len(set(compounds))
    ages = [r["tyreAge"] for r in rows if r["tyreAge"] is not None]
    if ages:
        out["maxTyreAgeLaps"] = max(ages)
        out["avgTyreAgeLaps"] = round(sum(ages) / len(ages), 2)
    if driver_stints:
        lengths = [s["end"] - s["start"] + 1 for s in driver_stints]
        out["maxStintLaps"] = max(lengths)
        out["avgStintLaps"] = round(sum(lengths) / len(lengths), 2)
    durations = [s["stopSeconds"] for s in stops if s["stopSeconds"] is not None]
    if durations:
        out["pitLaneTotalSeconds"] = round(sum(durations), 2)
        out["pitLaneMaxSeconds"] = round(max(durations), 2)
        out["pitLaneAvgSeconds"] = round(sum(durations) / len(durations), 2)
    positions = [r["pos"] for r in rows if r["pos"] is not None]
    if positions:
        out["positionStart"] = positions[0]
        out["positionEnd"] = positions[-1]
        out["positionMin"] = min(positions)
        out["positionMax"] = max(positions)
    return out


def build_feature_tables(session, base: dict, origin: float, t_end: float,
                        laps: dict, stints: dict, pit_stops: dict,
                        track_status: list[dict], race_control: list[dict],
                        weather: list[dict], drivers: list[dict],
                        stats_by_driver: dict, track) -> dict:
    """Per-session feature tables.  Every row carries the join keys
    (sessionKey / year / round / sessionType / driverNumber) so the tables can be
    merged into one flat dataset by build_features.py."""
    tables: dict[str, list[dict]] = {name: [] for name in FEATURE_TABLES}
    results = session.results
    res_by_abbr = {str(r["Abbreviation"]): r for _, r in results.iterrows()}
    res_by_number = {str(r["DriverNumber"]): r for _, r in results.iterrows()}
    team_by_abbr = {a: str(r.get("TeamName") or "") for a, r in res_by_abbr.items()}

    # ---- laps (one row per lap, with speed traps and the weather at that time)
    # one pass over the FastF1 frame instead of a boolean filter per lap row
    lap_extra: dict[tuple[str, int], dict] = {}
    for row in session.laps.itertuples(index=False):
        raw = row._asdict()
        lap_num = raw.get("LapNumber")
        if lap_num is None or pd.isna(lap_num):
            continue
        lap_extra[(str(raw.get("Driver")), int(lap_num))] = raw

    for abbr, rows in laps.items():
        number = str(res_by_abbr.get(abbr, {}).get("DriverNumber", ""))
        for r in rows:
            extra = lap_extra.get((abbr, r["lap"]))
            wx = _weather_at(weather, r["t0"] + origin)
            tables["laps"].append({
                "sessionKey": base["sessionKey"], "year": base["year"],
                "round": base["round"], "sessionType": base["sessionType"],
                "circuit": base["circuit"], "country": base["country"],
                "date": base["date"], "eventFormat": base["eventFormat"],
                "driverNumber": number, "driver": abbr, "teamName": team_by_abbr.get(abbr, ""),
                "lap": r["lap"], "lapTimeMs": r["time"],
                "sector1Ms": r["s1"], "sector2Ms": r["s2"], "sector3Ms": r["s3"],
                "speedTrapI1Kmh": None if extra is None else _none_or(_num(extra.get("SpeedI1"))),
                "speedTrapI2Kmh": None if extra is None else _none_or(_num(extra.get("SpeedI2"))),
                "speedTrapFLKmh": None if extra is None else _none_or(_num(extra.get("SpeedFL"))),
                "speedTrapSTKmh": None if extra is None else _none_or(_num(extra.get("SpeedST"))),
                "compound": r["compound"], "tyreAgeLaps": r["tyreAge"],
                "position": r["pos"], "lapStartTimeSec": r["t0"],
                "pitInSec": r["pitIn"], "pitOutSec": r["pitOut"],
                "trackStatus": r["ts"], "isDeleted": r["del"],
                "isFastF1Generated": r["gen"], "isAccurate": r["acc"],
                "isPersonalBest": None if extra is None else bool(extra.get("IsPersonalBest") == True),
                "deletedReason": None if extra is None else (None if pd.isna(extra.get("DeletedReason"))
                                                             else str(extra.get("DeletedReason"))),
                "airTempC": _none_or(wx.get("airTemp")), "trackTempC": _none_or(wx.get("trackTemp")),
                "humidityPct": _none_or(wx.get("humidity")), "pressureHpa": _none_or(wx.get("pressure")),
                "rainfallMm": _none_or(wx.get("rainfall")), "windSpeedMs": _none_or(wx.get("windSpeed")),
                "windDirectionDeg": _none_or(wx.get("windDirection")),
            })

    # ---- stints
    for abbr, rows in stints.items():
        number = str(res_by_abbr.get(abbr, {}).get("DriverNumber", ""))
        lap_by_number = {r["lap"]: r for r in laps.get(abbr, [])}
        for i, s in enumerate(rows, 1):
            lap_rows = [lap_by_number[l] for l in range(s["start"], s["end"] + 1)
                        if l in lap_by_number]
            times = [r["time"] for r in lap_rows if r["time"] is not None]
            ages = [r["tyreAge"] for r in lap_rows if r["tyreAge"] is not None]
            tables["stints"].append({
                "sessionKey": base["sessionKey"], "year": base["year"],
                "round": base["round"], "sessionType": base["sessionType"],
                "circuit": base["circuit"], "driverNumber": number, "driver": abbr,
                "teamName": team_by_abbr.get(abbr, ""), "stint": i,
                "compound": s["compound"], "startLap": s["start"], "endLap": s["end"],
                "stintLaps": s["end"] - s["start"] + 1,
                "tyreAgeStartLaps": min(ages) if ages else None,
                "tyreAgeEndLaps": max(ages) if ages else None,
                "stintLapTimeTotalMs": sum(times) if times else None,
                "stintAvgLapMs": round(sum(times) / len(times), 1) if times else None,
                "stintBestLapMs": min(times) if times else None,
            })

    # ---- pit stops
    for abbr, stops in pit_stops.items():
        number = str(res_by_abbr.get(abbr, {}).get("DriverNumber", ""))
        for s in stops:
            tables["pit_stops"].append({
                "sessionKey": base["sessionKey"], "year": base["year"],
                "round": base["round"], "sessionType": base["sessionType"],
                "circuit": base["circuit"], "driverNumber": number, "driver": abbr,
                "teamName": team_by_abbr.get(abbr, ""), **s,
            })

    # ---- session results (race / qualifying classification + per-driver aggregates)
    for _, r in results.sort_values("Position").iterrows():
        abbr = str(r["Abbreviation"])
        agg = _driver_aggregates(laps.get(abbr, []), pit_stops.get(abbr, []),
                                 stints.get(abbr, []))
        tel = stats_by_driver.get(abbr, {})
        tables["session_results"].append({
            "sessionKey": base["sessionKey"], "year": base["year"],
            "round": base["round"], "sessionType": base["sessionType"],
            "eventName": base["eventName"], "circuit": base["circuit"],
            "country": base["country"], "date": base["date"],
            "eventFormat": base["eventFormat"], "sessionDurationSec": base["duration"],
            "totalLaps": base["totalLaps"], "trackLengthMeters": base["trackLengthMeters"],
            "driverNumber": str(r["DriverNumber"]), "driver": abbr,
            "driverId": None if pd.isna(r.get("DriverId")) else str(r["DriverId"]),
            "firstName": None if pd.isna(r.get("FirstName")) else str(r["FirstName"]),
            "lastName": None if pd.isna(r.get("LastName")) else str(r["LastName"]),
            "teamName": None if pd.isna(r.get("TeamName")) else str(r["TeamName"]),
            "teamId": None if pd.isna(r.get("TeamId")) else str(r["TeamId"]),
            "teamColor": None if pd.isna(r.get("TeamColor")) else str(r["TeamColor"]),
            "countryCode": None if pd.isna(r.get("CountryCode")) else str(r["CountryCode"]),
            "gridPosition": _int_or(r.get("GridPosition")),
            "finishingPosition": _int_or(r.get("Position")),
            "classifiedPosition": _int_or(r.get("ClassifiedPosition")),
            "status": None if pd.isna(r.get("Status")) else str(r["Status"]),
            "points": None if pd.isna(r.get("Points")) else float(r["Points"]),
            "lapsDeclared": _int_or(r.get("Laps")),
            "q1Ms": _ms(r.get("Q1")) if "Q1" in results.columns else None,
            "q2Ms": _ms(r.get("Q2")) if "Q2" in results.columns else None,
            "q3Ms": _ms(r.get("Q3")) if "Q3" in results.columns else None,
            **agg, **{k: v for k, v in tel.items() if k not in agg},
        })

    # ---- telemetry features (kept separate so they can be joined on their own)
    for abbr, stats in stats_by_driver.items():
        number = str(res_by_abbr.get(abbr, {}).get("DriverNumber", ""))
        tables["telemetry_stats"].append({
            "sessionKey": base["sessionKey"], "year": base["year"],
            "round": base["round"], "sessionType": base["sessionType"],
            "circuit": base["circuit"], "driverNumber": number, "driver": abbr,
            "teamName": team_by_abbr.get(abbr, ""), **stats,
        })

    # ---- driver_session_features: compact per-driver summary (the modelling table)
    for entry in drivers:
        abbr = entry["abbr"]
        agg = _driver_aggregates(laps.get(abbr, []), pit_stops.get(abbr, []),
                                 stints.get(abbr, []))
        tel = stats_by_driver.get(abbr, {})
        tables["driver_session_features"].append({
            "sessionKey": base["sessionKey"], "year": base["year"],
            "round": base["round"], "sessionType": base["sessionType"],
            "eventName": base["eventName"], "circuit": base["circuit"],
            "country": base["country"], "date": base["date"],
            "eventFormat": base["eventFormat"], "trackLengthMeters": base["trackLengthMeters"],
            "totalLaps": base["totalLaps"], "sessionDurationSec": base["duration"],
            **base["weatherSummary"], **base["trackStatusSummary"],
            "driverNumber": entry["number"], "driver": abbr,
            "teamName": entry["team"], "teamColor": entry["teamColor"],
            "gridPosition": entry["grid"], "finishingPosition": entry["finish"],
            "status": entry["status"], "points": entry["points"],
            "positionsGained": None if (entry["grid"] is None or entry["finish"] is None)
                                else entry["grid"] - entry["finish"],
            **agg, **{k: v for k, v in tel.items() if k not in agg},
        })

    # ---- track status / race control / weather
    for i, row in enumerate(track_status):
        nxt = track_status[i + 1]["t"] if i + 1 < len(track_status) else None
        tables["track_status"].append({
            "sessionKey": base["sessionKey"], "year": base["year"],
            "round": base["round"], "sessionType": base["sessionType"],
            "circuit": base["circuit"], **row,
            "durationSeconds": None if nxt is None else round(nxt - row["t"], 2),
        })
    for row in race_control:
        tables["race_control"].append({
            "sessionKey": base["sessionKey"], "year": base["year"],
            "round": base["round"], "sessionType": base["sessionType"],
            "circuit": base["circuit"], **row,
        })
    for row in weather:
        tables["weather"].append({
            "sessionKey": base["sessionKey"], "year": base["year"],
            "round": base["round"], "sessionType": base["sessionType"],
            "circuit": base["circuit"], **row,
        })

    # ---- sessions (one row per session: the top-level feature table)
    counts, seconds = _aggregate_track_status(track_status)
    valid_laps = [(abbr, r) for abbr, rows in laps.items() for r in rows
                  if r["time"] is not None and not r["del"] and not r["gen"]]
    fastest = min(valid_laps, key=lambda p: p[1]["time"]) if valid_laps else None
    retired = [d for d in drivers if not str(d["status"]).startswith("Finished")]
    pole_driver, pole_ms = None, None
    if base["sessionType"] == "Q":
        top = [d for d in drivers if d["finish"] == 1]
        if top:
            pole_driver, pole_ms = top[0]["abbr"], (top[0].get("bestLapMs")
                                                    or stats_by_driver.get(top[0]["abbr"], {}).get("bestLapMs"))
    else:
        top = [d for d in drivers if d["grid"] == 1]
        if top:
            pole_driver = top[0]["abbr"]
    winner = next((d for d in drivers if d["finish"] == 1), None)
    tables["sessions"].append({
        "sessionKey": base["sessionKey"], "slug": base["slug"], "year": base["year"],
        "round": base["round"], "sessionType": base["sessionType"],
        "eventName": base["eventName"], "eventFormat": base["eventFormat"],
        "country": base["country"], "location": base["location"], "circuit": base["circuit"],
        "date": base["date"], "totalLaps": base["totalLaps"],
        "sessionDurationSec": base["duration"], "driverCount": len(drivers),
        "hasPosition": base["hasPosition"], "trackLengthMeters": base["trackLengthMeters"],
        "pitLaneMeters": base.get("pitLaneMeters"),
        "lapRows": sum(len(v) for v in laps.values()),
        "pitStopsTotal": sum(len(v) for v in pit_stops.values()),
        "stintCountTotal": sum(len(v) for v in stints.values()),
        "retiredCount": len(retired),
        "fastestLapMs": None if fastest is None else fastest[1]["time"],
        "fastestLapDriver": None if fastest is None else fastest[0],
        "fastestLapNumber": None if fastest is None else fastest[1]["lap"],
        "poleDriver": pole_driver, "poleLapMs": pole_ms,
        "winnerDriver": None if winner is None else winner["abbr"],
        "winnerTeam": None if winner is None else winner["team"],
        "raceControlMessages": len(race_control),
        "trackStatusEvents": len(track_status),
        **base["weatherSummary"],
        **{f"trackStatus_{k}Count": v for k, v in counts.items()},
        **{f"trackStatus_{k}Seconds": round(v, 1) for k, v in seconds.items()},
        "telemetrySamplesTotal": sum(s.get("telemetrySamples", 0) for s in stats_by_driver.values()),
        "generatedAt": base["generatedAt"], "source": base["source"],
    })
    return tables


def _none_or(value):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _int_or(value):
    """安全な整数変換。'R' のようなコードや NaN は None になる。"""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or not re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", value):
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number)


def _json_safe(obj):
    """Recursively make a structure strict-JSON compatible.

    json.dumps() writes bare `NaN` / `Infinity` tokens by default, which
    `JSON.parse()` in the browser rejects.  Everything non-finite becomes null.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


_JSON_DEFAULT_WARNED: set[str] = set()


def _json_default(obj):
    """Last-resort encoder for pandas/numpy objects that survived _json_safe()."""
    key = type(obj).__name__
    if key not in _JSON_DEFAULT_WARNED:
        _JSON_DEFAULT_WARNED.add(key)
        print(f"  ! json: unexpected type {key} encoded as text", flush=True)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def _dump_gz(obj, path: str) -> int:
    """Serialize as strict JSON and write it gzipped.  Returns the file size."""
    blob = json.dumps(_json_safe(obj), ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False, default=_json_default).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(gzip.compress(blob, 6))
    return os.path.getsize(path)


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
    has_position = track is not None
    if has_position:
        print(f"      {len(track['points'])} points from {track['referenceLaps']} reference lap(s), "
              f"{track['arcUnits']} units / {track['referenceLapMeters']} m "
              f"-> {track['unitMeters']:.4f} m per unit", flush=True)
    else:
        print("      no positional data for this session (FastF1 has none before 2018 round 3); "
              "exporting without the track map", flush=True)

    print("[3/6] driver telemetry series ...", flush=True)
    series: dict[str, dict] = {}
    stats_by_driver: dict[str, dict] = {}
    pit_points: list[tuple[float, float, float]] = []
    for i, abbr in enumerate(session.laps.groupby("Driver").groups.keys(), 1):
        built, pits, stats = build_driver(session, abbr, origin, t_end)
        series[abbr] = built
        stats_by_driver[abbr] = stats
        pit_points.extend(pits)
        print(f"      {i}: {abbr} ({len(built.get('x', []))} keyframes)", flush=True)

    pitlane = build_pitlane(pit_points, track) if has_position else []
    pit_lane_meters = None
    if len(pitlane) >= 2:
        # pitlane points are in exported 1/100 m units
        pit_lane_meters = round(
            sum(math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(pitlane, pitlane[1:])) / (XY_Q * XY_Q), 1)
    print(f"      pit lane points: {len(pitlane)}"
          + (f" / {pit_lane_meters} m" if pit_lane_meters else ""), flush=True)

    print("[4/6] lap tables / stints / pit stops ...", flush=True)
    laps = build_laps(session, origin)
    stints = build_stints(laps)
    pit_stops = build_pit_stops(laps)

    print("[5/6] race control / track status / weather ...", flush=True)
    track_status = build_track_status(session, origin)
    race_control = build_race_control(session, origin)
    weather = build_weather(session, origin)

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
            "grid": _int_or(r.get("GridPosition")),
            "finish": _int_or(r.get("Position")),
            "status": str(r["Status"]),
            "points": None if pd.isna(r["Points"]) else float(r["Points"]),
            "laps": _int_or(r.get("Laps")),
            "bestLapMs": None,
        })

    slug = f"{year}-r{int(round_number):02d}-{session_type.lower()}"
    event_format = str(session.event.get("EventFormat", ""))
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    track_length = None
    if has_position:
        track_length = round(track["arcUnits"] * track["unitMeters"], 1)
    base = {
        "slug": slug,
        "year": int(year),
        "round": int(round_number),
        "sessionType": session_type,
        "sessionLabel": SESSION_LABELS.get(session_type, session_type),
        "eventName": str(session.event["OfficialEventName"]),
        "eventFormat": event_format,
        "country": str(session.event["Country"]),
        "location": str(session.event["Location"]),
        "circuit": str(session.event["Location"]),
        "date": str(session.date.date()),
        "sessionKey": slug,
        "totalLaps": int(session.laps["LapNumber"].max() or 0),
        "duration": round(t_end, 1),
        "hasPosition": has_position,
        "trackLengthMeters": track_length,
        "pitLaneMeters": pit_lane_meters,
        "generatedAt": generated_at,
        "source": f"FastF1 {fastf1.__version__}",
    }
    base["weatherSummary"] = _weather_summary(weather)
    ts_counts, ts_seconds = _aggregate_track_status(track_status)
    base["trackStatusSummary"] = {
        "yellowFlagEvents": ts_counts.get("yellow", 0),
        "doubleYellowEvents": ts_counts.get("double_yellow", 0),
        "safetyCarEvents": ts_counts.get("safety_car", 0),
        "virtualSafetyCarEvents": ts_counts.get("virtual_safety_car", 0),
        "redFlagEvents": ts_counts.get("red", 0),
        "safetyCarSeconds": round(ts_seconds.get("safety_car", 0.0), 1),
        "virtualSafetyCarSeconds": round(ts_seconds.get("virtual_safety_car", 0.0), 1),
        "redFlagSeconds": round(ts_seconds.get("red", 0.0), 1),
    }
    for entry in drivers:
        # best lap of the session, ignoring deleted and FastF1 generated laps
        times = [r["time"] for r in laps.get(entry["abbr"], [])
                 if r["time"] is not None and not r["del"] and not r["gen"]]
        entry["bestLapMs"] = min(times) if times else None

    payload = {
        "meta": {
            **base,
            "dtMain": DT_MAIN,
            "dtDetail": DT_DETAIL,
            "relScale": REL_Q,
            "xyScale": XY_Q,
        },
        "track": None if not has_position else {
            "outline": track["points"], "pitlane": pitlane, "bbox": track["bbox"],
            "arcLength": track["arcLength"],
            "arcUnits": track["arcUnits"],
            "unitMeters": track["unitMeters"],
            "lengthMeters": track_length,
            "referenceLapMeters": track["referenceLapMeters"],
            "referenceLaps": track["referenceLaps"],
        },
        "drivers": drivers,
        "series": series,
        "laps": laps,
        "stints": stints,
        "pitStops": pit_stops,
        "trackStatus": track_status,
        "raceControl": race_control,
        "weather": weather,
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{slug}.json.gz")
    size = _dump_gz(payload, path)

    # feature tables (raw per-session JSON, gzipped; merged by build_features.py)
    tables = build_feature_tables(session, base, origin, t_end, laps, stints, pit_stops,
                                  track_status, race_control, weather, drivers,
                                  stats_by_driver, track)
    feat_dir = os.path.join(out_dir, "features", "raw")
    os.makedirs(feat_dir, exist_ok=True)
    feat_path = os.path.join(feat_dir, f"{slug}.json.gz")
    _dump_gz(tables, feat_path)

    # index part (so index.json can be rebuilt without re-reading every payload)
    part_dir = os.path.join(out_dir, "index-parts")
    os.makedirs(part_dir, exist_ok=True)
    with open(os.path.join(part_dir, f"{slug}.json"), "w", encoding="utf-8") as fh:
        json.dump(_json_safe(_index_entry(base, len(drivers), size, feat_path)), fh,
                  ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                  default=_json_default)

    counts = {name: len(rows) for name, rows in tables.items()}
    print(f"[6/6] wrote {path} ({size / 1024 / 1024:.2f} MB) + {feat_path} "
          f"({os.path.getsize(feat_path) / 1024:.0f} KB, {counts}) "
          f"in {time.time() - t_start:.1f}s", flush=True)
    return slug


def _index_entry(base: dict, n_drivers: int, size: int, feat_path: str) -> dict:
    return {
        "slug": base["slug"],
        "file": f"{base['slug']}.json.gz",
        "featuresFile": f"features/raw/{os.path.basename(feat_path)}",
        "order": SESSION_ORDER.get(base["sessionType"], 9),
        "year": base["year"], "round": base["round"],
        "sessionType": base["sessionType"], "sessionLabel": base["sessionLabel"],
        "eventFormat": base["eventFormat"],
        "label": f"{base['year']} R{base['round']:02d} {base['country']} ({base['sessionLabel']})",
        "eventName": base["eventName"], "date": base["date"],
        "country": base["country"], "location": base["location"], "circuit": base["circuit"],
        "drivers": n_drivers, "totalLaps": base["totalLaps"], "duration": base["duration"],
        "hasPosition": base["hasPosition"],
        "trackLengthMeters": base["trackLengthMeters"],
        "sizeKB": round(size / 1024),
    }


def _read_json_gz_or_plain(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def update_index(out_dir: str) -> None:
    entries: dict[str, dict] = {}
    part_dir = os.path.join(out_dir, "index-parts")
    if os.path.isdir(part_dir):
        for name in sorted(os.listdir(part_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(part_dir, name), encoding="utf-8") as fh:
                    entry = json.load(fh)
                entry["file"] = name[: -len(".json")] + ".json.gz"
                entries[entry["slug"]] = entry
            except Exception as exc:
                print(f"index: skipping part {name}: {exc}", flush=True)

    for name in sorted(os.listdir(out_dir)):
        if not (name.endswith(".json.gz") or name.endswith(".json")) or name == "index.json":
            continue
        slug = name[:-len(".json.gz")] if name.endswith(".json.gz") else name[:-len(".json")]
        if slug in entries:
            continue
        try:
            meta = _read_json_gz_or_plain(os.path.join(out_dir, name))["meta"]
        except Exception as exc:
            print(f"index: skipping {name}: {exc}", flush=True)
            continue
        entries[slug] = _index_entry({**meta,
                                      "sessionLabel": SESSION_LABELS.get(meta.get("sessionType"),
                                                                        meta.get("sessionType", ""))},
                                     len(meta.get("drivers", [])),
                                     os.path.getsize(os.path.join(out_dir, name)),
                                     os.path.join(out_dir, "features", "raw", f"{slug}.json.gz"))

    ordered = sorted(entries.values(),
                     key=lambda e: (e["year"], e["round"], e["order"], e["slug"]))
    index = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(ordered),
        "sessions": ordered,
    }
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(_json_safe(index), fh, ensure_ascii=False, indent=1,
                  allow_nan=False, default=_json_default)
    print(f"index: {len(ordered)} session(s) -> {os.path.join(out_dir, 'index.json')}", flush=True)


# ---------------------------------------------------------------- bulk mode
def enumerate_sessions(years: list[int], types: list[str]) -> list[tuple[int, int, str]]:
    """Every Race / Sprint / Sprint Qualifying / Qualifying session of `years`."""
    fastf1.Cache.enable_cache(".fastf1-cache")
    today = datetime.now(timezone.utc).date()
    tasks: list[tuple[int, int, str]] = []
    for year in years:
        schedule = fastf1.get_event_schedule(year)
        for _, ev in schedule.iterrows():
            try:
                round_number = int(ev["RoundNumber"])
            except (TypeError, ValueError):
                continue
            if round_number <= 0 or str(ev.get("EventFormat")) == "testing":
                continue
            if bool(ev.get("F1ApiSupport", True)) is False:
                continue
            try:
                event_date = pd.Timestamp(ev["EventDate"]).date()
            except Exception:
                event_date = None
            if event_date is not None and event_date > today:
                continue
            codes: list[str] = []
            for key in ("Session1", "Session2", "Session3", "Session4", "Session5"):
                name = ev.get(key)
                if name is None or pd.isna(name):
                    continue
                code = SESSION_NAME_TO_CODE.get(str(name))
                if code and code in types and code not in codes:
                    codes.append(code)
            for code in codes:
                tasks.append((year, round_number, code))
    return tasks


def _parse_years(spec: str) -> list[int]:
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def _output_paths(out_dir: str, task: tuple[int, int, str]) -> tuple[str, str]:
    year, round_number, code = task
    slug = f"{year}-r{int(round_number):02d}-{code.lower()}"
    return (os.path.join(out_dir, f"{slug}.json.gz"),
            os.path.join(out_dir, "features", "raw", f"{slug}.json.gz"))


def build_safe(task: tuple[int, int, str], out_dir: str, attempts: int = 6) -> str:
    """Build one session, retrying transient failures.

    FastF1 enforces a client side rate limit (500 API calls per rolling hour),
    so a bulk run that starts from an empty cache hits it.  Those errors need a
    much longer wait than a cache-lock error, otherwise the retries just burn
    more of the same budget.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return build(task[0], task[1], task[2], out_dir)
        except Exception as exc:  # requests-cache lock / transient API errors
            last = exc
            text = f"{type(exc).__name__}: {exc}"
            rate_limited = "RateLimit" in text or "calls/h" in text
            wait = (90.0 + 45.0 * i) if rate_limited else (2.0 + 4.0 * i)
            print(f"    retry {i + 1}/{attempts} {task} in {wait:.0f}s: {text[:140]}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{task[0]} R{task[1]} {task[2]}: {type(last).__name__}: {last}") from last


def run_bulk(args) -> int:
    years = _parse_years(args.years)
    types = [t.strip().upper() for t in args.types.split(",") if t.strip()]
    print(f"bulk: years={years} types={types}", flush=True)
    tasks = enumerate_sessions(years, types)
    print(f"bulk: {len(tasks)} candidate sessions", flush=True)

    todo = []
    skipped = 0
    for task in tasks:
        data_path, feat_path = _output_paths(args.out, task)
        if not args.force and os.path.exists(data_path) and os.path.exists(feat_path):
            skipped += 1
            continue
        todo.append(task)
    print(f"bulk: {len(todo)} to build, {skipped} already present", flush=True)
    if args.limit:
        todo = todo[: args.limit]
        print(f"bulk: limited to first {len(todo)}", flush=True)
    if args.dry_run:
        for task in todo:
            print("  would build", task, flush=True)
        return 0

    done: list[str] = []
    failures: list[dict] = []
    t0 = time.time()
    if args.workers <= 1:
        for i, task in enumerate(todo, 1):
            print(f"--- {i}/{len(todo)} {task}", flush=True)
            try:
                done.append(build_safe(task, args.out))
            except Exception as exc:
                failures.append({"task": list(task), "error": str(exc)})
                print(f"!!! failed {task}: {exc}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(build_safe, task, args.out): task for task in todo}
            for i, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    done.append(future.result())
                    print(f"--- {i}/{len(todo)} done {task}", flush=True)
                except Exception as exc:
                    failures.append({"task": list(task), "error": str(exc)})
                    print(f"!!! {i}/{len(todo)} failed {task}: {exc}", flush=True)

    update_index(args.out)
    with open(os.path.join(args.out, "build_failures.json"), "w", encoding="utf-8") as fh:
        json.dump({"failedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "failures": failures}, fh, ensure_ascii=False, indent=1)
    print(f"bulk: {len(done)} built, {len(failures)} failed in {time.time() - t0:.0f}s",
          flush=True)
    return 0 if not failures else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="FastF1 -> static JSON + feature export")
    ap.add_argument("--year", type=int)
    ap.add_argument("--round", type=int)
    ap.add_argument("--session", default="R", help="R / Q / S / SQ / FP1 ...")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "data"))
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--bulk", action="store_true", help="export every session of --years")
    ap.add_argument("--years", default="2018-2026", help="e.g. 2018-2026 or 2024,2025")
    ap.add_argument("--types", default="R,Q,S,SQ", help="session types for --bulk")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.bulk:
        return run_bulk(args)
    if args.year is None or args.round is None:
        ap.error("--year and --round are required unless --bulk is used")
    build(args.year, args.round, args.session.upper(), args.out)
    if not args.no_index:
        update_index(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
