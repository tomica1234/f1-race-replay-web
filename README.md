# F1 Race Replay Web

[FastF1](https://docs.fastf1.dev/) で F1 のセッションデータをダウンロードし、

1. **任意のセッションのリプレイ（コース上の車両アニメーション）＋分析グラフ**を見る Web アプリ、および
2. **機械学習の特徴量として使えるテーブル（CSV / Parquet）** の書き出し

をまとめて行うリポジトリです。Python で変換して `data/` に置き、GitHub Pages にそのまま公開します（Pages 側で Python は動きません）。

公開先: https://tomica1234.github.io/f1-race-replay-web/

## できること

- **2018 年以降の全セッションを時系列で閲覧**（決勝 / 予選 / スプリント / スプリント予選）。年・種別で絞り込み、`◀ ▶` で前後のセッションへ移動できます。
- リプレイ（再生 / 1〜60 倍速 / シーク / ラップ送り / 順位表 / コース状態バッジ）と分析グラフ（ラップタイム・順位推移・スティント・速度/DRS・レースコントロール）。
- **データ出力タブ**から、読み込み中のセッションの表を CSV でその場にダウンロード、または全セッション合本の機械学習用テーブル（CSV / Parquet）をダウンロード。列の意味（日本語）も同じ画面に一覧表示されます。

## 画面

| パネル | 内容 |
| --- | --- |
| 上部 | セッション選択（年 → 大会 → 種別でグループ化）、年・種別の絞り込み、`◀ ▶` で前後のセッションへ移動 |
| 左（リプレイ） | 全車の位置をテレメトリ実測座標で描画。再生 / 速度 1〜60 倍 / シーク / ラップ送り、ラップカウンター、コース状態バッジ（GREEN / YELLOW / SC / VSC / RED） |
| 右（リーダーボード） | 現在の順位、周回数、リーダーとの差（秒換算）、周回遅れ、ピット表示、タイヤコンパウンド、リタイア表示 |
| 下（分析） | 6 タブ: ラップタイム / 順位推移 / スティント / テレメトリ（速度・ギア・DRS・スロットル・ブレーキ） / レースコントロール / **データ出力** |

車両クリック・リーダーボードクリック・ドライバーチップで表示する車両を選択できます。
キー操作: `Space` 再生/停止、`←` `→` ラップ送り、`Esc` 全選択解除。

## リポジトリ構成

```
index.html / app.js / style.css   # アプリ本体（依存は vanilla JS / Canvas のみ、CDN 不使用）
build_data.py                     # FastF1 → セッション JSON + 特徴量 JSON（1 セッション = 1 ファイル）
build_features.py                 # 特徴量 JSON → 卓別 CSV / Parquet + schema.json（日本語列辞書）
data/
  index.json                      # セッション一覧（アプリが最初に読む。時系列順）
  <slug>.json.gz                  # リプレイ用ペイロード（gzip、1 セッション 0.3〜2.3 MB）
  index-parts/<slug>.json         # index.json を組み立てるためのメタ（ペイロードを読み直さないため）
  features/
    raw/<slug>.json.gz            # 1 セッション分の生特徴量（10 卓）
    <table>.csv / <table>.parquet # 全セッション合本（機械学習用）
    schema.json                   # 全列の日本語の説明・単位・型
  build_failures.json             # 生成に失敗したセッションとその理由
.fastf1-cache/                    # FastF1 のキャッシュ（git 管理外）
.nojekyll                         # GitHub Pages の Jekyll 処理を止める（静的ファイルそのまま配信）
```

GitHub Pages はリポジトリ直下（`main` ブランチの `/`）を配信しています。Pages の配信元は `/` か `/docs` に限定されるため、アプリのファイルはリポジトリ直下に置いています。

## データの生成

Python 3.12 と FastF1（および Parquet 出力に pyarrow）が必要です。

```bash
uv venv --python 3.12 && .venv/bin/python -m pip install fastf1 pyarrow   # または python3.12 -m venv .venv

# 1 セッションだけ生成する
.venv/bin/python build_data.py --year 2026 --round 16 --session R

# 2018 年以降の全セッション（決勝・予選・スプリント・スプリント予選）をまとめて生成
.venv/bin/python build_data.py --bulk --workers 5

# 特徴量テーブル（CSV / Parquet / schema.json）を合成
.venv/bin/python build_features.py
```

`build_data.py --bulk` は**中断しても続きから再開できます**（出力済みのセッションはスキップ）。`--force` で全作り直し、`--years 2024-2026`、`--types R,Q`、`--limit 20`、`--dry-run` で対象を絞れます。

FastF1 には **クライアント側のレート制限（500 API 呼び出し/時）**があるため、キャッシュが空の状態から全セッションを生成すると数時間かかります。`.fastf1-cache/` がある状態での再生成はネットワークを使いません（`build_data.py` はレート制限を検知して自動で待ちます）。

## 機械学習用の特徴量テーブル

`build_features.py` が `data/features/` に書き出す卓（すべて CSV と Parquet の両方、`schema.json` に全列の日本語の説明があります）:

| 卓 | 1 行の意味 | 主な列 |
| --- | --- | --- |
| `sessions` | 1 セッション | 大会情報、1 周距離、ピットレーン距離、天気の集計、フラグ/セーフティカーの回数と時間、ファステラップ、ポール、優勝 |
| `session_results` | セッション × ドライバー（公式結果） | グリッド、最終順位、ポイント、Q1/Q2/Q3、ラップ・スティント・ピットの集計、テレメトリ統計 |
| `driver_session_features` | セッション × ドライバー（**主表**） | 上の表に天気・コース状態の文脈特徴を結合したもの |
| `laps` | 1 周回 | ラップタイム、セクター、速度トラップ、タイヤ、順位、その時点の天気 |
| `stints` | 1 スティント | 使用タイヤ、スティント長、スティント中の平均/最速ラップ |
| `pit_stops` | 1 ピットストップ | テレメトリから復元したピットイン/アウト時刻と停止時間 |
| `telemetry_stats` | セッション × ドライバー | 速度・スロットル・ブレーキ・DRS・ギア・加減速の分布統計 |
| `track_status` / `race_control` / `weather` | 時系列イベント | 時刻（セッション開始基準）、状態、メッセージ、気温・路面温度・湿度・気圧・風・降雨 |
| `tracks` / `drivers` / `teams` | 次元テーブル（導出） | コース属性、ドライバー通算成績（出走・優勝・表彰台・ポール・ポイント）、チーム同 |

結合キーは `sessionKey`（= ファイルの slug、例 `2024-r05-r`）で、`driverNumber` / `lap` / `stint` / `stop` と組み合わせます。

読み込み例:

```python
import pandas as pd
driver = pd.read_parquet("data/features/driver_session_features.parquet")
laps = pd.read_parquet("data/features/laps.parquet")
# CSV は UTF-8（BOM 付き）なので Excel でも文字化けしません
```

## データサイズと GitHub Pages について

- ペイロードは gzip 圧縮（圧縮率 約 0.31）で保存し、**選択したセッションだけをその場で読み込みます**（`DecompressionStream('gzip')`。Chrome 100 / Safari 16.4 以降）。index.json は数 MB のメタのみです。
- 全セッション分のリプレイ JSON は生で約 2 GB、gzip で 600 MB 前後、特徴量の CSV/Parquet 合本は 100 MB 前後になります。GitHub Pages の「1 サイト 1 GB 目安」「1 か月 100 GB 転送」に収まる構成にしています（超える場合は `data/` を分割するか、リプレイの解像度を落とす必要があります）。

## 実装上の注意（FastF1 の性質に由来）

- テレメトリの `X/Y` は GPS ではなく F1 のタイミングシステム座標です（1 単位 = 1/10 m）。地図ではなく回路の形状として描画しています。
- 周回ごとの `RelativeDistance` は各周回で位相がずれるため、コース輪郭は「基準周回 1 周」から作り、開始点と長さが近い周回だけ平均化しています。
- **2018 年ラウンド 1・2 には位置データがありません**（FastF1 が保持していない）。これらのセッションは `hasPosition: false` で出力され、アプリではコース図を出さず「位置データなし」を表示します。順位・ラップ・スティント・ピット・テレメトリのデータと分析タブは使えます。
- ピット停止は `Status` が終始 `OnTrack` なので、`PitInTime`〜`PitOutTime` の間のテレメトリからピットレーンを復元しています。FastF1 は `PitOutTime` を翌周回の行に記録します。タイミングシステムがピット内で車両を追跡していないセッション（予選に多い）ではピットレーンを出しません（`pitlane: []`、`pitLaneMeters: null`）。
- `pitLaneMeters` は **F1 のタイミング定義（ピットレーン入口線〜出口線）に沿った走行距離**です。建物内の区間ではなく入口ランプ・出口ランプを含むため、図面のピットレーンより長く出ます（同じ回路でもセッションによりばらつきます）。厳密な施設長が必要な場合は `pit_stops.pitIn/pitOut` と `stopSeconds`、`pitLaneAvgSeconds` の方を使ってください。
- セーフティカーの位置テレメトリは存在しません（上流の f1-race-replay は先頭車の約 500 m 前に SC を描画しています）。このアプリは状態バッジとコース状態バーで表現します。
- テレメトリが取得できるのは 2018 年以降です。`isFastF1Generated`（計測から補間された周回）と `isDeleted`（公式記録から除外された周回）はフラグとして保持し、集計からは除いています。
- 順位は `(LapNumber - 1) + RelativeDistance` をクライアント側で計算したライブ順位で、公式最終結果とは数値が異なる場合があります。

## 派生元・ライセンス

- データと発想は [IAmTomShaw/f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) と [FastF1](https://github.com/theOehrly/Fast-F1) に基づきます。
- F1 公式データおよび Formula 1 の名称・ロゴは Formula One Licensing B.V. の商標です。個人利用・非商用のデモとして公開しています。
