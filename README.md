# Stock Analysis & Monitoring System (J-Stock)

[English](#english) | [日本語](#japanese)

<a name="english"></a>
## English

### Overview
This is a high-performance stock analysis and monitoring system designed for the Japanese stock market (JPX). It automatically fetches daily stock prices, performs technical analysis using vectorized logic for extreme speed, and detects stocks showing strong buy signals based on multiple technical indicators.

### Key Features
*   **High-Speed Data Fetching**: Uses smart grouping to fetch only missing data (differential update) from Yahoo Finance, minimizing network usage and time.
*   **Vectorized Analysis**: Implements technical analysis logic using Pandas vectorization, achieving **25x speedup** compared to traditional loop-based methods (approx. 18 seconds for ~3800 stocks).
*   **Signal Detection**: Scores stocks (0-9) based on **9 technical criteria**:
    1.  Price above MA150 & MA200
    2.  MA150 > MA200 (Golden Cross state)
    3.  MA200 trending upwards
    4.  Higher Highs & Higher Lows (Uptrend)
    5.  Volume increase during uptrends
    6.  Weekly volume analysis
    7.  **RSI < 70** (Avoid overbought)
    8.  **MACD Golden Cross** (Trend reversal)
    9.  **Trading Value Filter** (> 100M JPY)
*   **Backtesting & Exit Strategy**:
    *   Simulates past trades to calculate win rates and returns.
    *   **Exit Strategy**: Simulates "Take Profit at +10%" and "Stop Loss at -5%".
    *   **Proven Performance**: Score 9 stocks showed **+0.53% average return per trade** with this strategy.
*   **Report Generation**: Automatically generates `latest_report.txt` with backtest results and a list of **Recommended Entry/Target/Stop-Loss Prices**.
*   **Dockerized**: Fully containerized environment for easy deployment.

### Prerequisites
*   Docker & Docker Compose

### Usage

1.  **Start the container**
    ```bash
    docker-compose up -d
    ```

2.  **Run the Monitor**
    This single command fetches the latest data, analyzes all stocks, and generates a report.
    ```bash
    docker-compose exec app python run_monitor.py
    ```

3.  **Check Results**
    *   **`latest_report.txt`**: Contains backtest statistics and a list of recommended stocks with specific **Target Price** and **Stop Loss**.
    *   **`analysis_results.csv`**: Full analysis results for all stocks.

---

<a name="japanese"></a>
## 日本語

### 概要
日本株（JPX）を対象とした高性能な株価分析・監視システムです。日々の株価データを自動取得し、ベクトル化された高速ロジックを用いてテクニカル分析を行い、強い買いシグナルが出ている銘柄を検出します。

### 主な機能
*   **高速データ取得**: スマートグルーピング機能により、不足している期間（差分）のデータのみをYahoo Financeから一括取得します。
*   **爆速分析ロジック**: Pandasのベクトル化演算を駆使し、ループ処理を排除。約3800銘柄の全期間分析を **約18秒** で完了します（従来比約25倍）。
*   **シグナル検出**: 以下の **9つのテクニカル指標** に基づき、銘柄をスコアリング（0〜9点）します。
    1.  株価が150日・200日移動平均線より上にある
    2.  150日線 > 200日線（パーフェクトオーダーに近い状態）
    3.  200日線が上昇傾向
    4.  高値・安値の切り上げ（ダウ理論的上昇トレンド）
    5.  上昇時の出来高増・下落時の出来高減
    6.  週足での出来高分析
    7.  **RSI < 70**（買われすぎによる高値掴みを回避）
    8.  **MACDゴールデンクロス**（トレンドの初動を捕捉）
    9.  **売買代金フィルター**（1日1億円以上、流動性を確保）
*   **バックテスト＆出口戦略**:
    *   過去のデータを用いて「もし条件を満たした日に買っていたら？」をシミュレーション。
    *   **出口戦略**: 「利確+10%、損切-5%」のルールで運用した場合の勝率とリターンを算出。
    *   **実績**: Score 9の銘柄で平均リターン **+0.53%** を記録（バックテスト結果）。
*   **レポート自動生成**: バックテスト結果と、推奨銘柄の **「利確価格」「損切価格」** を記載した `latest_report.txt` を自動生成します。
*   **Docker対応**: 環境構築不要で、すぐに利用可能です。

### 動作環境
*   Docker & Docker Compose

### 使い方

1.  **コンテナの起動**
    ```bash
    docker-compose up -d
    ```

2.  **監視システムの実行**
    以下のコマンド一つで、データ更新から分析、レポート作成まで全自動で行います。
    ```bash
    docker-compose exec app python run_monitor.py
    ```

3.  **結果の確認**
    *   **`latest_report.txt`**: バックテストの詳細結果と、推奨銘柄リスト（利確・損切価格付き）が保存されます。文字化けの心配もありません。
    *   **`analysis_results.csv`**: 全銘柄の詳細な分析結果が保存されます。
