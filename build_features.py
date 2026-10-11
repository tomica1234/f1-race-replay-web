#!/usr/bin/env python3
"""Merge the per-session feature exports into machine-learning tables.

`build_data.py` writes one gzipped JSON per session under
``data/features/raw/<sessionKey>.json.gz``.  This script merges all of them into
one table per feature group and writes each group twice:

    data/features/<table>.csv      (UTF-8 with BOM, opens cleanly in Excel)
    data/features/<table>.parquet  (zstd, for pandas / polars / sklearn)

It also derives three small dimension tables (``tracks``, ``drivers``, ``teams``)
and writes ``data/features/schema.json``, a Japanese description of every column
so the export is usable without reading the source code.

Usage:
    python build_features.py                 # merge everything in data/features/raw
    python build_features.py --no-parquet    # CSV only
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import statistics
import sys
from datetime import datetime, timezone

import pandas as pd

try:
    import pyarrow  # noqa: F401
    HAVE_PYARROW = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_PYARROW = False


# ---------------------------------------------------------------------------
# table metadata (Japanese)
# ---------------------------------------------------------------------------

TABLE_LABELS: dict[str, tuple[str, str, list[str]]] = {
    "sessions": (
        "セッション単位（1セッション = 1行）",
        "大会・天候・コース・レースコントロールの集計。セッション単位の説明変数はここから取る。",
        ["sessionKey"],
    ),
    "session_results": (
        "セッション結果 × ドライバー（公式結果 + ラップ/テレメトリ集計）",
        "公式タイムシートにドライバー単位の集計特徴を結合した表。目的変数（順位・ポイント）もここにある。",
        ["sessionKey", "driverNumber"],
    ),
    "driver_session_features": (
        "ドライバー × セッション（天気・フラグ集計を含む横持ち特徴表）",
        "機械学習用の主表。セッション文脈（天気・セーフティカー・赤旗）とドライバー行動特徴が同じ行に揃う。",
        ["sessionKey", "driverNumber"],
    ),
    "laps": (
        "ラップ単位（1周 = 1行）",
        "ラップタイム・セクター・タイヤ・順位・その時点の天気。ラップ単位のモデルを作る場合に使う。",
        ["sessionKey", "driverNumber", "lap"],
    ),
    "stints": (
        "スティント単位（1スティント = 1行）",
        "同一タイヤの連続走行区間。スティント長とペースの関係を分析できる。",
        ["sessionKey", "driverNumber", "stint"],
    ),
    "pit_stops": (
        "ピットストップ単位（1回 = 1行）",
        "テレメトリから復元したピットイン/アウト時刻と停止時間。",
        ["sessionKey", "driverNumber", "stop"],
    ),
    "telemetry_stats": (
        "ドライバー × セッションのテレメトリ統計",
        "速度・スロットル・ブレーキ・DRS・ギア・加減速の分布統計（driver_session_features の内訳）。",
        ["sessionKey", "driverNumber"],
    ),
    "track_status": (
        "トラックステータス変更（グリーン/イエロー/セーフティカー等）",
        "セッション中のコース状態変化の時系列。",
        ["sessionKey", "t"],
    ),
    "race_control": (
        "レースコントロールメッセージ",
        "公式メッセージログ（時刻・周回・対象ドライバー）。",
        ["sessionKey", "t"],
    ),
    "weather": (
        "天気サンプル（セッション時系列）",
        "FastF1 が取得した気温・路面温度・湿度・気圧・風・降雨の時系列。",
        ["sessionKey", "t"],
    ),
    "tracks": (
        "コース次元テーブル（1サーキット = 1行）",
        "セッション表から導出したコース属性。周回数やコース長の統計を含む。",
        ["circuit"],
    ),
    "drivers": (
        "ドライバー次元テーブル（1ドライバー = 1行）",
        "セッション結果表から導出した通算成績（出走・優勝・表彰台・ポール・ポイント）。",
        ["driver"],
    ),
    "teams": (
        "チーム次元テーブル（1チーム = 1行）",
        "セッション結果表から導出したチームの通算成績。",
        ["teamName"],
    ),
}

# column -> (日本語名, 単位・補足)
COLUMN_LABELS: dict[str, tuple[str, str]] = {
    # --- 結合キー / 大会情報 ---
    "sessionKey": ("セッションキー", "全テーブルの外部キー。data/<slug>.json.gz の slug と同一"),
    "slug": ("データファイルのスラッグ", "例 2024-r05-r"),
    "year": ("年", "シーズン"),
    "round": ("ラウンド番号", "その年の大会番号"),
    "sessionType": ("セッション種別", "R=決勝 / Q=予選 / S=スプリント / SQ=スプリント予選"),
    "sessionLabel": ("セッション種別（日本語）", "決勝 / 予選 / スプリント / スプリント予選"),
    "eventName": ("大会公式名称", "例 FORMULA 1 ... GRAND PRIX"),
    "eventFormat": ("大会フォーマット", "conventional / sprint / sprint_qualifying"),
    "country": ("開催国", ""),
    "location": ("開催地", "FastF1 の Location（通常はサーキット名）"),
    "circuit": ("コース", "location と同じ値。次元テーブルのキー"),
    "date": ("開催日", "YYYY-MM-DD"),
    "totalLaps": ("規定周回数", "周"),
    "sessionDurationSec": ("セッション所要時間", "秒（計測開始から最終サンプルまで）"),
    "driverCount": ("出走ドライバー数", "人"),
    "hasPosition": ("位置データの有無", "true のとき X/Y 座標とコース形状が得られる"),
    "trackLengthMeters": ("1周距離", "メートル（テレメトリの X/Y から算出。GPS ではない）"),
    "pitLaneMeters": ("ピットレーン距離", "メートル（テレメトリから算出）"),
    "lapRows": ("ラップ行数", "FastF1 が記録したラップ数"),
    "pitStopsTotal": ("ピットストップ総数", "回"),
    "stintCountTotal": ("スティント総数", "回"),
    "retiredCount": ("リタイア人数", "人"),
    "fastestLapMs": ("ファステラップ", "ミリ秒"),
    "fastestLapDriver": ("ファステラップ記録者", "ドライバー略称"),
    "fastestLapNumber": ("ファステラップの周回", "周"),
    "poleDriver": ("ポールポジション獲得者", "決勝の Q3 最速タイム記録者"),
    "poleLapMs": ("ポールタイム", "ミリ秒"),
    "winnerDriver": ("優勝ドライバー", ""),
    "winnerTeam": ("優勝チーム", ""),
    "raceControlMessages": ("レースコントロールメッセージ数", "件"),
    "trackStatusEvents": ("トラックステータス変更回数", "回"),
    "telemetrySamplesTotal": ("テレメトリサンプル総数", "全ドライバー合計"),
    "generatedAt": ("生成日時", "UTC ISO8601"),
    "source": ("データソース", "FastF1 のバージョン"),

    # --- 天気 ---
    "weatherSamples": ("天気サンプル数", "件"),
    "avgAirTempC": ("気温（平均）", "℃"),
    "minAirTempC": ("気温（最小）", "℃"),
    "maxAirTempC": ("気温（最大）", "℃"),
    "avgTrackTempC": ("路面温度（平均）", "℃"),
    "minTrackTempC": ("路面温度（最小）", "℃"),
    "maxTrackTempC": ("路面温度（最大）", "℃"),
    "avgHumidityPct": ("湿度（平均）", "%"),
    "minHumidityPct": ("湿度（最小）", "%"),
    "maxHumidityPct": ("湿度（最大）", "%"),
    "avgPressureHpa": ("気圧（平均）", "hPa"),
    "minPressureHpa": ("気圧（最小）", "hPa"),
    "maxPressureHpa": ("気圧（最大）", "hPa"),
    "avgWindSpeedMs": ("風速（平均）", "m/s"),
    "minWindSpeedMs": ("風速（最小）", "m/s"),
    "maxWindSpeedMs": ("風速（最大）", "m/s"),
    "rainfallTotalMm": ("降雨量の合計", "mm"),
    "rainfallSamples": ("降雨が観測されたサンプル数", "件"),
    "isDry": ("ドライコンディション", "降雨サンプルが 0 なら true"),
    "airTempC": ("気温", "℃（その時点の値）"),
    "trackTempC": ("路面温度", "℃（その時点の値）"),
    "humidityPct": ("湿度", "%（その時点の値）"),
    "pressureHpa": ("気圧", "hPa（その時点の値）"),
    "rainfallMm": ("降雨量", "mm（その時点の値）"),
    "windSpeedMs": ("風速", "m/s（その時点の値）"),
    "windDirectionDeg": ("風向", "度（その時点の値）"),

    # --- トラックステータス / レースコントロール ---
    "trackStatus_greenCount": ("グリーン状態の回数", "回"),
    "trackStatus_yellowCount": ("イエロー状態の回数", "回"),
    "trackStatus_double_yellowCount": ("ダブルイエローの回数", "回"),
    "trackStatus_redCount": ("赤旗の回数", "回"),
    "trackStatus_safety_carCount": ("セーフティカー出動回数", "回"),
    "trackStatus_virtual_safety_carCount": ("バーチャルセーフティカー回数", "回"),
    "trackStatus_greenSeconds": ("グリーン状態の合計時間", "秒"),
    "trackStatus_yellowSeconds": ("イエロー状態の合計時間", "秒"),
    "trackStatus_double_yellowSeconds": ("ダブルイエローの合計時間", "秒"),
    "trackStatus_redSeconds": ("赤旗の合計時間", "秒"),
    "trackStatus_safety_carSeconds": ("セーフティカー走行時間", "秒"),
    "trackStatus_virtual_safety_carSeconds": ("バーチャルセーフティカー時間", "秒"),
    "trackStatus_otherCount": ("その他の状態値（上記以外）の回数", "回"),
    "trackStatus_otherSeconds": ("その他の状態値の合計時間", "秒"),
    "yellowFlagEvents": ("イエローフラグ表示回数", "回"),
    "doubleYellowEvents": ("ダブルイエロー表示回数", "回"),
    "safetyCarEvents": ("セーフティカー出動回数", "回"),
    "virtualSafetyCarEvents": ("バーチャルセーフティカー回数", "回"),
    "redFlagEvents": ("赤旗表示回数", "回"),
    "safetyCarSeconds": ("セーフティカー走行時間", "秒"),
    "virtualSafetyCarSeconds": ("バーチャルセーフティカー時間", "秒"),
    "redFlagSeconds": ("赤旗による中断時間", "秒"),
    "t": ("セッション開始からの時刻", "秒（0 =  lights out / セッション開始）"),
    "rel": ("セッション開始からの時刻", "秒"),
    "code": ("トラックステータスコード", "FastF1 のコード 1..7"),
    "name": ("トラックステータス名", "green / yellow / red / safety_car など"),
    "label": ("トラックステータス（日本語）", "グリーン / イエロー / セーフティカー など"),
    "durationSeconds": ("状態の継続時間", "秒"),
    "lap": ("周回番号", "周"),
    "flag": ("フラグ", "公式メッセージのフラグ種別"),
    "category": ("メッセージ種別", "FastF1 が分類したカテゴリ"),
    "driver": ("ドライバー略称", "3文字（例 VER）"),
    "text": ("メッセージ本文", ""),
    "airTemp": ("気温", "℃"),
    "trackTemp": ("路面温度", "℃"),
    "humidity": ("湿度", "%"),
    "pressure": ("気圧", "hPa"),
    "rainfall": ("降雨量", "mm"),
    "windSpeed": ("風速", "m/s"),
    "windDirection": ("風向", "度"),

    # --- ドライバー / チーム ---
    "driverNumber": ("カーナンバー", "例 44"),
    "driverId": ("ドライバーID", "F1 公式 ID"),
    "firstName": ("名", ""),
    "lastName": ("姓", ""),
    "teamName": ("チーム名", ""),
    "teamId": ("チームID", "F1 公式 ID"),
    "teamColor": ("チームカラー", "16進 RGB（# なし）"),
    "countryCode": ("国籍コード", "F1 公式の 3 文字コード"),
    "abbr": ("ドライバー略称", "3文字"),
    "team": ("チーム名", ""),

    # --- 結果 ---
    "gridPosition": ("スターティンググリッド", "周回開始時のポジション"),
    "finishingPosition": ("最終順位", "公式順位"),
    "classifiedPosition": ("分類順位", "完走扱いの順位（非完走は null）"),
    "status": ("出走結果", "Finished / +1L / DNF など公式ステータス"),
    "points": ("獲得ポイント", "点"),
    "lapsDeclared": ("公式記録の周回数", "周"),
    "q1Ms": ("Q1 タイム", "ミリ秒"),
    "q2Ms": ("Q2 タイム", "ミリ秒"),
    "q3Ms": ("Q3 タイム", "ミリ秒"),
    "positionsGained": ("グリッドからの順位上昇", "gridPosition - finishingPosition"),

    # --- ラップ集計 ---
    "lapsCompleted": ("完走周回数", "周（ラップ行から集計）"),
    "pitStops": ("ピットストップ回数", "回"),
    "stintCount": ("スティント数", "回"),
    "bestLapMs": ("ベストラップ", "ミリ秒（FastF1 記録）"),
    "avgLapMs": ("平均ラップタイム", "ミリ秒"),
    "stddevLapMs": ("ラップタイムの標準偏差", "ミリ秒"),
    "lastLapMs": ("最終ラップタイム", "ミリ秒"),
    "bestLapValidMs": ("有効なベストラップ", "ミリ秒（削除・補完ラップを除く）"),
    "deletedLaps": ("削除ラップ数", "件"),
    "deletedLapMsSaved": ("削除ラップの合計時間", "ミリ秒"),
    "bestS1Ms": ("セクター1 ベスト", "ミリ秒"),
    "bestS2Ms": ("セクター2 ベスト", "ミリ秒"),
    "bestS3Ms": ("セクター3 ベスト", "ミリ秒"),
    "compoundsUsed": ("使用コンパウンド内訳", "例 SOFT:12,MEDIUM:5"),
    "compoundStintCount": ("コンパウンド別スティント数", "件"),
    "maxTyreAgeLaps": ("タイヤ使用周回の最大", "周"),
    "avgTyreAgeLaps": ("タイヤ使用周回の平均", "周"),
    "maxStintLaps": ("最長スティント", "周"),
    "avgStintLaps": ("平均スティント長", "周"),
    "pitLaneTotalSeconds": ("ピットレーン走行時間の合計", "秒"),
    "pitLaneMaxSeconds": ("ピットレーン走行時間の最大", "秒"),
    "pitLaneAvgSeconds": ("ピットレーン走行時間の平均", "秒"),
    "positionStart": ("セッション開始時の順位", "位"),
    "positionEnd": ("セッション終了時の順位", "位"),
    "positionMin": ("最順位", "位"),
    "positionMax": ("最高順位", "位"),

    # --- ラップ単位 ---
    "lapTimeMs": ("ラップタイム", "ミリ秒"),
    "sector1Ms": ("セクター1 タイム", "ミリ秒"),
    "sector2Ms": ("セクター2 タイム", "ミリ秒"),
    "sector3Ms": ("セクター3 タイム", "ミリ秒"),
    "speedTrapI1Kmh": ("速度トラップ I1", "km/h"),
    "speedTrapI2Kmh": ("速度トラップ I2", "km/h"),
    "speedTrapFLKmh": ("速度トラップ（フィニッシュライン）", "km/h"),
    "speedTrapSTKmh": ("速度トラップ（最終計測点）", "km/h"),
    "compound": ("タイヤコンパウンド", "SOFT / MEDIUM / HARD / INTERMEDIATE / WET"),
    "tyreAgeLaps": ("タイヤ使用周回", "周（ラップ開始時点）"),
    "position": ("順位", "位（そのラップ時点）"),
    "lapStartTimeSec": ("ラップ開始時刻", "秒（セッション開始基準）"),
    "pitInSec": ("ピットイン時刻", "秒（セッション開始基準）"),
    "pitOutSec": ("ピットアウト時刻", "秒（セッション開始基準）"),
    "trackStatus": ("トラックステータス", "その時点のコード"),
    "isDeleted": ("削除ラップ", "true のとき公式記録から除外されている"),
    "isFastF1Generated": ("FastF1 補完ラップ", "計測データから補間して生成されたラップ"),
    "isAccurate": ("計測精度", "FastF1 がラップタイムを正確と判断したか"),
    "isPersonalBest": ("本人のベスト", "そのドライバーの最良ラップか"),
    "deletedReason": ("削除理由", "FastF1 の削除理由"),

    # --- スティント ---
    "stint": ("スティント番号", "1 始まり"),
    "startLap": ("スティント開始周回", "周"),
    "endLap": ("スティント終了周回", "周"),
    "stintLaps": ("スティント周回数", "周"),
    "tyreAgeStartLaps": ("スティント開始時のタイヤ使用周回", "周"),
    "tyreAgeEndLaps": ("スティント終了時のタイヤ使用周回", "周"),
    "stintLapTimeTotalMs": ("スティント合計ラップタイム", "ミリ秒"),
    "stintAvgLapMs": ("スティント平均ラップタイム", "ミリ秒"),
    "stintBestLapMs": ("スティント中ベストラップ", "ミリ秒"),

    # --- ピットストップ ---
    "stop": ("ピットストップ番号", "ドライバー内で 1 始まり"),
    "lapBefore": ("ピット前の周回", "周"),
    "pitIn": ("ピットイン時刻", "秒（セッション開始基準）"),
    "pitOut": ("ピットアウト時刻", "秒（セッション開始基準）"),
    "stopSeconds": ("ピット停止時間", "秒（ピットインからアウトまで）"),
    "lapTimeBeforeMs": ("ピット前のラップタイム", "ミリ秒"),
    "compoundBefore": ("ピット前のコンパウンド", ""),
    "tyreAgeBefore": ("ピット前のタイヤ使用周回", "周"),

    # --- テレメトリ統計 ---
    "telemetrySamples": ("テレメトリサンプル数", "件（詳細系列 2.0 秒間隔）"),
    "telemetrySeconds": ("テレメトリ計測時間", "秒"),
    "speedMaxKmh": ("最高速度", "km/h"),
    "speedMinKmh": ("最低速度", "km/h"),
    "speedMeanKmh": ("平均速度", "km/h"),
    "speedP95Kmh": ("速度の95パーセンタイル", "km/h"),
    "speedOver300Seconds": ("300km/h 超で走行した時間", "秒"),
    "speedOver250Seconds": ("250km/h 超で走行した時間", "秒"),
    "speedOver200Seconds": ("200km/h 超で走行した時間", "秒"),
    "speedOver100Seconds": ("100km/h 超で走行した時間", "秒"),
    "throttleMeanPct": ("スロットル開度（平均）", "%"),
    "throttleP95Pct": ("スロットル開度（95パーセンタイル）", "%"),
    "throttleMaxPct": ("スロットル開度（最大）", "%"),
    "throttleOver50Seconds": ("スロットル50%超の時間", "秒"),
    "throttleZeroSeconds": ("スロットル0%の時間", "秒"),
    "brakeSeconds": ("ブレーキ使用時間", "秒"),
    "brakeRatioPct": ("ブレーキ使用率", "%（計測時間に対する割合）"),
    "drsZoneSeconds": ("DRSゾーン検知時間", "秒"),
    "drsOpenSeconds": ("DRS開放時間", "秒"),
    "drsOpenRatioPct": ("DRS開放率", "%（DRSゾーン時間に対する割合）"),
    "gearMax": ("最高ギア", ""),
    "gearMin": ("最低ギア", ""),
    "gearChanges": ("ギアチェンジ回数", "回（サンプル間の変化数）"),
    "accelMaxMs2": ("最大加速度", "m/s^2"),
    "accelP95Ms2": ("加速度の95パーセンタイル", "m/s^2"),
    "decelMaxMs2": ("最大減速度", "m/s^2"),
    "decelP05Ms2": ("減速度の5パーセンタイル", "m/s^2"),
    "stationarySeconds": ("停止状態の時間", "秒"),
    "offTrackSeconds": ("コース外走行の時間", "秒"),
    "onTrackSeconds": ("コース内走行の時間", "秒"),

    # --- 次元テーブル（導出） ---
    "firstYear": ("初登場年", ""),
    "lastYear": ("最終登場年", ""),
    "years": ("登場年数", "年"),
    "sessionCount": ("出走セッション数", "件"),
    "raceCount": ("決勝出走数", "件"),
    "sprintCount": ("スプリント出走数", "件"),
    "qualifyingCount": ("予選出走数", "件"),
    "wins": ("優勝数", "回"),
    "podiums": ("表彰台回数", "回"),
    "poles": ("ポールポジション数", "回"),
    "pointsTotal": ("獲得ポイント合計", "点"),
    "retirements": ("リタイア数", "回"),
    "teams": ("在籍チーム", "出現したチーム名（時系列順）"),
    "numbers": ("使用カーナンバー", "出現したナンバー"),
    "drivers": ("出走ドライバー", "出現したドライバー略称（時系列順）"),
    "fastestLapMsMin": ("コース最速ラップ記録", "ミリ秒"),
    "circuitCount": ("出走コース数", "件"),
    "avgTotalLaps": ("規定周回数の平均", "周"),
    "trackLengthMetersMedian": ("1周距離の中央値", "メートル"),
    "pitLaneMetersMedian": ("ピットレーン距離の中央値", "メートル"),
    "pitStopsAvgPerRace": ("決勝1戦あたり平均ピット回数", "回"),
    "bestLapMsMin": ("最速ラップ記録", "ミリ秒"),
    "speedMaxKmhMax": ("最高速度の最大", "km/h"),
    "driverCount": ("出走ドライバー数", "人"),
    "events": ("開催大会数", "件"),
}

DTYPE_NAMES = {
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "object": "string",
    "datetime": "timestamp",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_raw(path: str) -> dict:
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _collect(raw_dir: str) -> dict[str, tuple[list[dict], list[str]]]:
    """Merge every raw per-session feature file into rows + column order per table."""
    merged: dict[str, tuple[list[dict], list[str]]] = {}
    files = sorted(glob.glob(os.path.join(raw_dir, "*.json.gz")))
    for path in files:
        try:
            data = _read_raw(path)
        except Exception as exc:
            print(f"skip {path}: {exc}", flush=True)
            continue
        for name, rows in data.items():
            if not isinstance(rows, list):
                continue
            rows_acc, cols = merged.setdefault(name, ([], []))
            for row in rows:
                for key in row:
                    if key not in cols:
                        cols.append(key)
            rows_acc.extend(rows)
    return merged


def _frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    # keep nulls as nulls, but let pandas pick numeric dtypes where possible
    for col in df.columns:
        series = df[col]
        if series.dtype == object:
            values = [v for v in series.tolist() if v is not None]
            if values and all(isinstance(v, bool) for v in values):
                df[col] = series.astype("object")
            elif values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                df[col] = pd.to_numeric(series, errors="coerce")
    return df


def _dtype_name(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    return "string"


def _column_spec(df: pd.DataFrame) -> list[dict]:
    spec = []
    for col in df.columns:
        label, unit = COLUMN_LABELS.get(col, (col, ""))
        spec.append({"name": col, "type": _dtype_name(df[col]), "label": label, "unit": unit})
    return spec


# ---------------------------------------------------------------------------
# derived dimension tables
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column, or an all-null column when an older export lacks it."""
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def _derive_tracks(sessions: pd.DataFrame) -> pd.DataFrame:
    if sessions.empty:
        return pd.DataFrame()
    rows = []
    for circuit, grp in sessions.groupby("circuit", dropna=False):
        lengths = [v for v in _col(grp, "trackLengthMeters").tolist() if isinstance(v, (int, float))]
        pits = [v for v in _col(grp, "pitLaneMeters").tolist() if isinstance(v, (int, float))]
        races = grp[_col(grp, "sessionType") == "R"]
        rows.append({
            "circuit": circuit,
            "location": _col(grp, "location").iloc[0],
            "country": _col(grp, "country").iloc[0],
            "firstYear": int(_col(grp, "year").min()),
            "lastYear": int(_col(grp, "year").max()),
            "years": int(_col(grp, "year").nunique()),
            "events": int(grp[["year", "round"]].drop_duplicates().shape[0]),
            "sessionCount": int(len(grp)),
            "avgTotalLaps": round(float(_col(grp, "totalLaps").mean()), 2),
            "trackLengthMetersMedian": round(statistics.median(lengths), 1) if lengths else None,
            "pitLaneMetersMedian": round(statistics.median(pits), 1) if pits else None,
            "pitStopsAvgPerRace": round(float(races["pitStopsTotal"].mean()), 2) if len(races) else None,
            "fastestLapMsMin": min((v for v in _col(grp, "fastestLapMs").tolist() if isinstance(v, (int, float))),
                                   default=None),
            "avgAirTempC": round(float(_col(grp, "avgAirTempC").mean()), 2)
            if _col(grp, "avgAirTempC").notna().any() else None,
        })
    return pd.DataFrame(rows).sort_values(["country", "circuit"]).reset_index(drop=True)


def _derive_drivers(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    for driver, grp in results.groupby("driver", dropna=False):
        races = grp[_col(grp, "sessionType") == "R"]
        sprints = grp[_col(grp, "sessionType") == "S"]
        quals = grp[_col(grp, "sessionType") == "Q"]
        fin = _col(races, "finishingPosition")
        points = [v for v in _col(races, "points").tolist() if isinstance(v, (int, float))]
        identities = grp.assign(driverId=_col(grp, "driverId"),
                               firstName=_col(grp, "firstName"),
                               lastName=_col(grp, "lastName"),
                               countryCode=_col(grp, "countryCode")) \
            .dropna(subset=["driverId"]).drop_duplicates(
                subset=["driverId", "firstName", "lastName"])
        first = identities.iloc[0] if len(identities) else grp.iloc[0]
        rows.append({
            "driver": driver,
            "driverId": first.get("driverId"),
            "firstName": first.get("firstName"),
            "lastName": first.get("lastName"),
            "countryCode": first.get("countryCode"),
            "firstYear": int(_col(grp, "year").min()),
            "lastYear": int(_col(grp, "year").max()),
            "years": int(_col(grp, "year").nunique()),
            "sessionCount": int(len(grp)),
            "raceCount": int(len(races)),
            "sprintCount": int(len(sprints)),
            "qualifyingCount": int(len(quals)),
            "wins": int((fin == 1).sum()),
            "podiums": int(((fin >= 1) & (fin <= 3)).sum()),
            "poles": int((_col(grp, "gridPosition") == 1).sum()),
            "pointsTotal": round(float(sum(points)), 1),
            "retirements": int(sum(1 for v in _col(races, "status").tolist()
                                   if isinstance(v, str) and not v.startswith("Finished"))),
            "teams": ",".join(dict.fromkeys(str(v) for v in _col(grp, "teamName").dropna().tolist())),
            "numbers": ",".join(dict.fromkeys(str(v) for v in _col(grp, "driverNumber").dropna().tolist())),
            "circuitCount": int(_col(grp, "circuit").nunique()),
            "bestLapMsMin": min((v for v in _col(grp, "bestLapMs").tolist() if isinstance(v, (int, float))),
                                default=None),
            "speedMaxKmhMax": max((v for v in _col(grp, "speedMaxKmh").tolist() if isinstance(v, (int, float))),
                                  default=None),
        })
    return pd.DataFrame(rows).sort_values(["lastYear", "pointsTotal", "driver"],
                                          ascending=[False, False, True]).reset_index(drop=True)


def _derive_teams(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    for team, grp in results.groupby("teamName", dropna=False):
        races = grp[_col(grp, "sessionType") == "R"]
        fin = _col(races, "finishingPosition")
        points = [v for v in _col(races, "points").tolist() if isinstance(v, (int, float))]
        colors = [v for v in _col(grp, "teamColor").dropna().tolist()]
        rows.append({
            "teamName": team,
            "teamId": _col(grp, "teamId").dropna().iloc[0] if _col(grp, "teamId").notna().any() else None,
            "teamColor": colors[0] if colors else None,
            "firstYear": int(_col(grp, "year").min()),
            "lastYear": int(_col(grp, "year").max()),
            "years": int(_col(grp, "year").nunique()),
            "sessionCount": int(len(grp)),
            "raceCount": int(len(races)),
            "wins": int((fin == 1).sum()),
            "podiums": int(((fin >= 1) & (fin <= 3)).sum()),
            "pointsTotal": round(float(sum(points)), 1),
            "retirements": int(sum(1 for v in _col(races, "status").tolist()
                                   if isinstance(v, str) and not v.startswith("Finished"))),
            "drivers": ",".join(dict.fromkeys(str(v) for v in _col(grp, "driver").dropna().tolist())),
        })
    return pd.DataFrame(rows).sort_values(["lastYear", "pointsTotal", "teamName"],
                                          ascending=[False, False, True]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="merge per-session feature exports into ML tables")
    parser.add_argument("--raw", default="data/features/raw", help="directory of raw per-session JSON")
    parser.add_argument("--out", default="data/features", help="output directory")
    parser.add_argument("--no-parquet", action="store_true", help="skip the parquet files")
    parser.add_argument("--no-csv", action="store_true", help="skip the CSV files")
    parser.add_argument("--tables", default="", help="comma separated subset of tables to write")
    args = parser.parse_args()

    if not os.path.isdir(args.raw):
        print(f"raw feature directory not found: {args.raw}", file=sys.stderr)
        return 1
    if not args.no_parquet and not HAVE_PYARROW:
        print("pyarrow is not installed; writing CSV only", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    merged = _collect(args.raw)
    if not merged:
        print("no feature files found", file=sys.stderr)
        return 1

    frames: dict[str, pd.DataFrame] = {}
    for name, (rows, columns) in merged.items():
        frames[name] = _frame(rows, columns)

    # derived dimension tables
    if "sessions" in frames:
        frames["tracks"] = _derive_tracks(frames["sessions"])
    if "session_results" in frames:
        frames["drivers"] = _derive_drivers(frames["session_results"])
        frames["teams"] = _derive_teams(frames["session_results"])

    wanted = [t.strip() for t in args.tables.split(",") if t.strip()]
    schema = {"generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "source": "build_data.py feature tables + derived dimension tables",
              "notes": [
                  "時刻はすべてセッション開始（フォーメーションライトアウト）からの秒。負値は開始前の走行。",
                  "X/Y 座標は GPS ではなく計測システム由来の 1/10 メートル単位。hasPosition=false のセッションには位置データがない。",
                  "isFastF1Generated=true のラップは計測から補間されたもの。isDeleted=true は公式記録から除外されたラップ。",
                  "テレメトリ詳細系列は 2.0 秒間隔、位置系列は 0.5 秒間隔から集計している。",
              ],
              "tables": {}}

    only = set(wanted) or set(frames)
    total_rows = 0
    for name in sorted(frames, key=lambda n: (n not in TABLE_LABELS, n)):
        df = frames[name]
        if name not in only or df.empty:
            continue
        label, description, primary = TABLE_LABELS.get(name, (name, "", []))
        entry = {"label": label, "description": description, "primaryKey": primary,
                 "rowCount": int(len(df)), "columns": _column_spec(df)}
        if not args.no_csv:
            csv_path = os.path.join(args.out, f"{name}.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            entry["csv"] = os.path.basename(csv_path)
            entry["csvBytes"] = os.path.getsize(csv_path)
        if not args.no_parquet and HAVE_PYARROW:
            pq_path = os.path.join(args.out, f"{name}.parquet")
            df.to_parquet(pq_path, engine="pyarrow", compression="zstd")
            entry["parquet"] = os.path.basename(pq_path)
            entry["parquetBytes"] = os.path.getsize(pq_path)
        schema["tables"][name] = entry
        total_rows += len(df)
        sizes = " + ".join(f"{p} ({b / 1024 / 1024:.2f} MB)"
                           for p, b in ((entry.get("csv"), entry.get("csvBytes", 0)),
                                        (entry.get("parquet"), entry.get("parquetBytes", 0))) if p)
        print(f"{name:24s} {len(df):7d} rows  {sizes}", flush=True)

    unknown = sorted({c["name"] for entry in schema["tables"].values()
                      for c in entry["columns"] if c["label"] == c["name"]})
    if unknown:
        print(f"unlabelled columns: {unknown}", flush=True)

    with open(os.path.join(args.out, "schema.json"), "w", encoding="utf-8") as fh:
        json.dump(schema, fh, ensure_ascii=False, indent=1)
    print(f"schema -> {os.path.join(args.out, 'schema.json')} "
          f"({len(schema['tables'])} tables, {total_rows} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
