import os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
from datetime import datetime, timedelta
import concurrent.futures
import time

# 警告を無視
warnings.filterwarnings('ignore')

DATA_DIR = 'data'

def load_data(file_path):
    """CSVファイルを読み込み、前処理を行う"""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Close'])
        df = df.sort_values('Date')
        df = df.reset_index(drop=True)
        
        return df
    except Exception:
        return None

def calculate_vectorized_conditions(df, params):
    """
    全期間の条件判定を一括で行う（ベクトル化）
    戻り値: 各条件のBoolean DataFrame, スコアSeries
    """
    window = params.get('trend_window', 10)
    
    # --- 1. 移動平均線 ---
    df['MA150'] = df['Close'].rolling(window=150).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['MA200_Slope'] = df['MA200'].diff(5)
    
    # Cond 1: 株価 > MA150 & MA200
    c1 = (df['Close'] > df['MA150']) & (df['Close'] > df['MA200'])
    
    # Cond 2: MA150 > MA200
    c2 = df['MA150'] > df['MA200']
    
    # Cond 3: MA200上昇
    c3 = df['MA200_Slope'] > 0
    
    # --- Cond 5: 出来高トレンド (Rolling) ---
    # 上昇日の出来高平均 > 下落日の出来高平均 (過去50日)
    # ベクトル化:
    # 1. 上昇日、下落日の出来高を抽出（他はNaNまたは0）
    # 2. rolling().mean() で計算
    
    price_diff = df['Close'].diff()
    vol_up = df['Volume'].where(price_diff > 0)
    vol_down = df['Volume'].where(price_diff < 0)
    
    # min_periods=1 でNaNがあっても計算するようにする
    avg_vol_up = vol_up.rolling(window=50, min_periods=10).mean()
    avg_vol_down = vol_down.rolling(window=50, min_periods=10).mean()
    
    c5 = avg_vol_up > avg_vol_down
    
    # --- Cond 6: 週足分析 (Resample & Broadcast) ---
    # 週足データを作成
    weekly_df = df.set_index('Date').resample('W').agg({
        'Close': 'last', 'Volume': 'sum'
    })
    weekly_diff = weekly_df['Close'].diff()
    weekly_vol_avg = weekly_df['Volume'].rolling(window=26, min_periods=5).mean()
    
    # 高出来高週の判定
    is_high_vol = weekly_df['Volume'] > weekly_vol_avg
    
    # 上昇・下落週の判定
    is_up = weekly_diff > 0
    is_down = weekly_diff < 0
    
    # 過去26週における「高出来高かつ上昇」の回数 vs 「高出来高かつ下落」の回数
    # rolling().sum() でカウント
    count_up_high_vol = (is_high_vol & is_up).astype(int).rolling(window=26, min_periods=5).sum()
    count_down_high_vol = (is_high_vol & is_down).astype(int).rolling(window=26, min_periods=5).sum()
    
    weekly_c6 = count_up_high_vol >= count_down_high_vol
    
    # 日足に戻す (reindex & ffill)
    # weeklyのindexは日曜などになっているため、日足のindexに合わせてffillする
    c6 = weekly_c6.reindex(df.set_index('Date').index, method='ffill').reset_index(drop=True)
    
    # --- Cond 4: トレンド (Higher Highs/Lows) ---
    # argrelextrema を使ってピークを検出
    # order=window なので、ピークが確定するのは window 日後。
    # なので、shift(window) して「確定したピーク」として扱う。
    
    # ピーク検出 (未来のデータを使わないように注意が必要だが、argrelextremaは局所的なので、
    # 「その時点でピークと判定された」ことの再現には window 分の遅延を入れるのが正解)
    
    # 全期間でピークを探す
    high_peaks_idx = argrelextrema(df['High'].values, np.greater_equal, order=window)[0]
    low_peaks_idx = argrelextrema(df['Low'].values, np.less_equal, order=window)[0]
    
    # ピークの値をSeriesにする（ピーク日のみ値があり、他はNaN）
    high_peaks = pd.Series(np.nan, index=df.index)
    high_peaks.iloc[high_peaks_idx] = df['High'].iloc[high_peaks_idx]
    
    low_peaks = pd.Series(np.nan, index=df.index)
    low_peaks.iloc[low_peaks_idx] = df['Low'].iloc[low_peaks_idx]
    
    # ピーク確定日（window日後）に値をシフト
    # これにより「今日時点で知っている最新のピーク」を作れる
    confirmed_high_peaks = high_peaks.shift(window)
    confirmed_low_peaks = low_peaks.shift(window)
    
    # 最新のピーク(last)と、その前のピーク(prev)を取得
    # ffill() で「直近のピーク」を埋める
    last_high = confirmed_high_peaks.ffill()
    # last_high が更新されたタイミング（＝新しいピークが確定した日）の直前の値を prev とする
    # つまり、last_high が変化した場所を探す
    
    # ちょっと工夫：
    # validなピーク値だけを取り出したSeriesを作る
    valid_high_idxs = confirmed_high_peaks.dropna().index
    valid_low_idxs = confirmed_low_peaks.dropna().index
    
    # ピーク間の比較結果（今回 > 前回）を計算
    # これを「ピーク確定日」にマッピングする
    
    # High
    if len(valid_high_idxs) > 1:
        v_highs = confirmed_high_peaks.dropna()
        high_trend_ok = v_highs > v_highs.shift(1) # 今回 > 前回
        # これを元の時系列に戻す
        s_high_trend = pd.Series(False, index=df.index)
        s_high_trend.loc[high_trend_ok.index] = high_trend_ok
        # ffill: 「新しいピークが来るまでは、前のトレンド状態を維持」
        # （厳密には「トレンド継続中」とみなす）
        c4_high = s_high_trend.replace(False, np.nan).ffill().fillna(False)
    else:
        c4_high = pd.Series(False, index=df.index)

    # Low
    if len(valid_low_idxs) > 1:
        v_lows = confirmed_low_peaks.dropna()
        low_trend_ok = v_lows > v_lows.shift(1)
        s_low_trend = pd.Series(False, index=df.index)
        s_low_trend.loc[low_trend_ok.index] = low_trend_ok
        c4_low = s_low_trend.replace(False, np.nan).ffill().fillna(False)
    else:
        c4_low = pd.Series(False, index=df.index)
        
    c4 = c4_high & c4_low
    
    # --- スコア計算 ---
    # Booleanをintに変換して合計
    score_series = (c1.astype(int) + c2.astype(int) + c3.astype(int) + 
                    c4.astype(int) + c5.astype(int) + c6.fillna(False).astype(int))
    
    return {
        'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5, 'c6': c6,
        'score': score_series
    }

def process_ticker(ticker, params):
    """1銘柄を処理する関数（ベクトル化版）"""
    try:
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        df = load_data(file_path)
        
        if df is None or len(df) < 200:
            return None

        # ベクトル化計算
        # 全期間のスコアが一気に返ってくる
        vec_res = calculate_vectorized_conditions(df, params)
        score_series = vec_res['score']
        
        # 最新の状態
        current_score = score_series.iloc[-1]
        
        # 条件: スコア3以上
        if current_score >= 3:
            current_price = df.iloc[-1]['Close']
            last_date = df.iloc[-1]['Date']
            date_str = last_date.strftime('%Y-%m-%d')
            
            # シグナル点灯日を探す（爆速）
            # スコアが3以上になっている連続区間の開始日を探す
            
            # 1. スコア < 3 の場所を探す
            mask_below = score_series < 3
            
            # 2. 直近から過去に遡って、最初に mask_below が True になる場所を探す
            # last_valid_index は遅いので、逆順にして idxmax を使うなどのテクニック
            
            # 逆順にする
            rev_mask = mask_below.iloc[::-1]
            
            # True（3未満）になっている最初のインデックス
            if rev_mask.any():
                # idxmaxは最初のTrueのindexを返す
                last_below_idx = rev_mask.idxmax()
                # その翌日がスタート日
                # indexは整数indexではなくDateになっているわけではない（reset_indexしてるので整数）
                start_idx = last_below_idx + 1
                if start_idx >= len(df):
                    start_idx = len(df) - 1 # 境界値ケア
            else:
                # ずっと3以上だった場合（稀だが）
                start_idx = 0
                
            signal_start = df.iloc[start_idx]['Date']
            start_str = signal_start.strftime('%Y-%m-%d')
            
            # 最新の条件詳細
            met_conditions = []
            for k in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']:
                if vec_res[k].iloc[-1]:
                    met_conditions.append(k.replace('c', 'cond'))
            
            return {
                'Ticker': ticker,
                'Price': current_price,
                'Date': date_str,
                'Signal_Start': start_str,
                'Score': current_score,
                'Met_Conditions': met_conditions
            }
    except Exception:
        return None
    return None

def main():
    print("分析を開始します（爆速ベクトル化モード）...")
    start_time = time.time()
    
    files = os.listdir(DATA_DIR)
    tickers = [f.replace(".csv", "") for f in files if f.endswith(".csv")]
    print(f"対象銘柄数: {len(tickers)}")
    
    best_params = {'trend_window': 15}
    results = []
    
    # 並列処理も併用して最強の速度を目指す
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_ticker, ticker, best_params) for ticker in tickers]
        
        completed_count = 0
        total_count = len(tickers)
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            
            completed_count += 1
            if completed_count % 1000 == 0:
                print(f"進捗: {completed_count}/{total_count}")

    results.sort(key=lambda x: x['Score'], reverse=True)
    
    elapsed = time.time() - start_time
    print(f"\n分析完了 (所要時間: {elapsed:.1f}秒)")
    print(f"条件を3つ以上満たした銘柄: {len(results)} 件")
    
    if results:
        res_df_top = pd.DataFrame(results[:20])
        display_cols = ['Ticker', 'Price', 'Date', 'Signal_Start', 'Score', 'Met_Conditions']
        print(res_df_top[display_cols])
        
        res_df_all = pd.DataFrame(results)
        res_df_all.to_csv("analysis_results.csv", index=False)
        print(f"全 {len(results)} 件の結果を analysis_results.csv に保存しました。")
    else:
        print("条件を満たす銘柄はありませんでした。")

if __name__ == "__main__":
    main()
