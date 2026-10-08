# F1 Race Replay Web

[FastF1](https://docs.fastf1.dev/) で F1 のセッションデータをダウンロードし、
**レースリプレイ（コース上の車両アニメーション）＋分析グラフ**を見るための静的 Web アプリです。
Python で JSON に変換して `site/` に置き、GitHub Pages にそのまま公開します（Pages 側で Python は動きません）。

公開先: https://tomica1234.github.io/f1-race-replay-web/

## 画面

| パネル | 内容 |
| --- | --- |
| 左（リプレイ） | 全車の位置をテレメトリ実測座標で描画。再生 / 速度 1〜60 倍 / シーク / ラップ送り、ラップカウンター、コース状態バッジ（GREEN / YELLOW / SC / VSC / RED） |
| 右（リーダーボード） | 現在の順位、周回数、リーダーとの差（秒換算）、周回遅れ、ピット表示、タイヤコンパウンド、リタイア表示 |
| 下（分析） | 5 タブ: ラップタイム / 順位推移 / スティント / テレメトリ（速度・ギア・DRS・スロットル・ブレーキ） / レースコントロール（コース状態バー＋メッセージログ） |

車両クリック・リーダーボードクリック・ドライバーチップで表示する車両を選択できます。
キー操作: `Space` 再生/停止、`←` `→` ラップ送り、`Esc` 全選択解除。

## 使い方

1. 上部のセレクトでセッションを選ぶ（現在は 2026 R16 バーレーンGP in マレーシア / 決勝のみ）。
2. `▶ 再生` を押す。`4x` などを押すと早送り、スライダーで任意の時刻へ移動。
3. 分析タブを選び、`トップ10` などのチップやリーダーボード行でグラフに載せる車両を選ぶ。

## リポジトリ構成

```
build_data.py        # FastF1 → JSON 変換（Python）
site/
  index.html         # アプリ本体（依存は vanilla JS / Canvas のみ、CDN 不使用）
  app.js
  style.css
  data/
    index.json       # セッション一覧（アプリが読み込む）
    2026-r16-r.json  # 変換済みセッションデータ（約 7.6 MB）
```

## データの再生成

Python 3.12 と FastF1 が必要です（`uv` を使っています）。

```bash
python3.12 -m venv .venv            # または: uv venv --python 3.12
.venv/bin/pip install fastf1
.venv/bin/python build_data.py --year 2026 --round 16 --session R
# 例: 2024 モンコ・プリクオン（予選）を追加する
.venv/bin/python build_data.py --year 2024 --round 8 --session Q
```

`--out` で出力先、`--no-index` で `index.json` の更新をスキップできます。
`.fastf1-cache/` にキャッシュを作っておくと 2 回目以降はネットワークなしで再生成できます。

生成される JSON の主なキー:

- `meta` … イベント情報、セッション長、サンプリング間隔（主系列 0.5 s / 詳細 2.0 s）
- `track` … `outline`（コース輪郭 1200 点）と `pitlane`、`bbox`、`lengthMeters`
- `drivers` … 出走者、チームカラー、グリッド、最終結果
- `series` … 車両ごとの位置 `x,y`（1/10 m 単位）、`lap`、`rel`、状態 `st`、速度・ギア・DRS・スロットル・ブレーキ
- `laps` / `stints` … 周回タイム・セクター・タイヤ、スティント
- `trackStatus` / `raceControl` … コース状態とレースコントロールメッセージ（時刻はセッション開始基準、**周回開始前は負値**）

## 実装上の注意（FastF1 の性質に由来）

- テレメトリの `X/Y` は GPS ではなく F1 のタイミングシステム座標です（1 単位 = 1/10 m）。地図ではなく回路の形状として描画しています。
- 周回ごとの `RelativeDistance` は各周回で位相がずれるため、コース輪郭は「基準周回 1 周」から作り、開始点と長さが近い周回だけ平均化しています。
- ピット停止は `Status` が終始 `OnTrack` なので、`PitInTime`〜`PitOutTime` の間のテレメトリからピットレーンを復元しています。FastF1 は `PitOutTime` を翌周回の行に記録します。
- セーフティカーの位置テレメトリは存在しません（上流の f1-race-replay は先頭車の約 500 m 前に SC を描画しています）。このアプリは状態バッジとコース状態バーで表現します。
- テレメトリが取得できるのは 2018 年以降です。`IsAccurate` / `FastF1Generated` の周回はラップタイムタブに薄い印で区別しています。
- 順位は `(LapNumber - 1) + RelativeDistance` をクライアント側で計算したライブ順位で、公式最終結果とは数値が異なる場合があります。

## 派生元・ライセンス

- データと発想は [IAmTomShaw/f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) と [FastF1](https://github.com/theOehrly/Fast-F1) に基づきます。
- F1 公式データおよび Formula 1 の名称・ロゴは Formula One Licensing B.V. の商標です。個人利用・非商用のデモとして公開しています。
