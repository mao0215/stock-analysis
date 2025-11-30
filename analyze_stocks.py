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
        # 必要なカラムだけ読むことで高速化
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # CloseがNaNの行は削除
        df = df.dropna(subset=['Close'])
                
        df = df.sort_values('Date')
        df = df.reset_index(drop=True)
        
        return df
    except Exception:
        return None

def calculate_technical_indicators(df):
    """テクニカル指標を計算する"""
    df = df.copy()
    
    # 1. 移動平均線
    df['MA150'] = df['Close'].rolling(window=150).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # 2. MA200の傾き (5日間の変化)
    df['MA200_Slope'] = df['MA200'].diff(5)
    
    return df

def check_trend_higher_highs_lows(df, window=10):
    """条件4: 高値と安値の切り上げ"""
    recent_df = df.tail(250).copy()
    if len(recent_df) < window * 2:
        return False

    high_idxs = argrelextrema(recent_df['High'].values, np.greater_equal, order=window)[0]
    low_idxs = argrelextrema(recent_df['Low'].values, np.less_equal, order=window)[0]
    
    if len(high_idxs) < 2 or len(low_idxs) < 2:
        return False

    last_high = recent_df.iloc[high_idxs[-1]]['High']
    prev_high = recent_df.iloc[high_idxs[-2]]['High']
    last_low = recent_df.iloc[low_idxs[-1]]['Low']
    prev_low = recent_df.iloc[low_idxs[-2]]['Low']
    
    return last_high > prev_high and last_low > prev_low

def check_volume_trend(df):
    """条件5: 出来高分析（緩和版）"""
    recent_df = df.tail(50).copy()
    recent_df['Change'] = recent_df['Close'].diff()
    up_days = recent_df[recent_df['Change'] > 0]
    down_days = recent_df[recent_df['Change'] < 0]
    
    if len(up_days) == 0 or len(down_days) == 0:
        return False
        
    avg_vol_up = up_days['Volume'].mean()
    avg_vol_down = down_days['Volume'].mean()
    return avg_vol_up > avg_vol_down

def check_weekly_volume(df):
    """条件6: 週足分析（緩和版）"""
    weekly_df = df.set_index('Date').resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    })
    recent_weekly = weekly_df.tail(26).copy()
    recent_weekly['Change'] = recent_weekly['Close'].diff()
    avg_weekly_vol = recent_weekly['Volume'].mean()
    high_vol_weeks = recent_weekly[recent_weekly['Volume'] > avg_weekly_vol]
    
    up_weeks_high_vol = high_vol_weeks[high_vol_weeks['Change'] > 0]
    down_weeks_high_vol = high_vol_weeks[high_vol_weeks['Change'] < 0]
    
    return len(up_weeks_high_vol) >= len(down_weeks_high_vol)

def evaluate_stock(df, params):
    """指定されたパラメータで銘柄を評価する"""
    current = df.iloc[-1]
    cond1 = current['Close'] > current['MA150'] and current['Close'] > current['MA200']
    cond2 = current['MA150'] > current['MA200']
    cond3 = current['MA200_Slope'] > 0
    cond4 = check_trend_higher_highs_lows(df, window=params['trend_window'])
    cond5 = check_volume_trend(df)
    cond6 = check_weekly_volume(df)
    
    conditions = [cond1, cond2, cond3, cond4, cond5, cond6]
    score = sum(conditions)
    
    return {
        'cond1': cond1, 'cond2': cond2, 'cond3': cond3, 
        'cond4': cond4, 'cond5': cond5, 'cond6': cond6,
        'score': score, 'all_met': all(conditions)
    }

def find_signal_start_date(df, params, threshold=3):
    """シグナル点灯日を探す"""
    max_days = 365
    current_idx = len(df) - 1
    start_date = df.iloc[current_idx]['Date']
    
    for i in range(1, max_days):
        idx = current_idx - i
        if idx < 200: break
        past_df = df.iloc[:idx+1]
        res = evaluate_stock(past_df, params)
        if res['score'] < threshold:
            return start_date
        start_date = df.iloc[idx]['Date']
    return start_date

def process_ticker(ticker, params):
    """1銘柄を処理する関数（並列実行用）"""
    try:
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        df = load_data(file_path)
        
        if df is None or len(df) < 200:
            return None

        # 指標計算
        df = calculate_technical_indicators(df)
        if len(df) < 200: return None
        
        # 評価
        res = evaluate_stock(df, params)
        
        # 条件: スコア3以上
        if res['score'] >= 3:
            current_price = df.iloc[-1]['Close']
            last_date = df.iloc[-1]['Date']
            date_str = last_date.strftime('%Y-%m-%d') if not pd.isna(last_date) else "Unknown"
            
            signal_start = find_signal_start_date(df, params, threshold=3)
            start_str = signal_start.strftime('%Y-%m-%d') if not pd.isna(signal_start) else "Unknown"
            
            return {
                'Ticker': ticker,
                'Price': current_price,
                'Date': date_str,
                'Signal_Start': start_str,
                'Score': res['score'],
                'Met_Conditions': [k for k, v in res.items() if k.startswith('cond') and v]
            }
    except Exception:
        return None
    return None

def main():
    print("分析を開始します...")
    start_time = time.time()
    
    files = os.listdir(DATA_DIR)
    tickers = [f.replace(".csv", "") for f in files if f.endswith(".csv")]
    print(f"対象銘柄数: {len(tickers)}")
    
    # パラメータ（固定）
    # 本来はチューニングすべきだが、高速化のため前回の最適値を使用
    best_params = {'trend_window': 15}
    
    results = []
    
    # 並列処理
    # CPUコア数に応じてプロセス数を自動調整
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # mapを使うと順序が保たれるが、今回は順序不問なのでas_completedでも良い
        # mapの方がシンプル
        futures = [executor.submit(process_ticker, ticker, best_params) for ticker in tickers]
        
        completed_count = 0
        total_count = len(tickers)
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            
            completed_count += 1
            if completed_count % 500 == 0:
                print(f"進捗: {completed_count}/{total_count}")

    # 結果処理
    results.sort(key=lambda x: x['Score'], reverse=True)
    
    print(f"\n分析完了 (所要時間: {time.time() - start_time:.1f}秒)")
    print(f"条件を3つ以上満たした銘柄: {len(results)} 件")
    
    if results:
        # コンソール表示用（上位20件）
        res_df_top = pd.DataFrame(results[:20])
        display_cols = ['Ticker', 'Price', 'Date', 'Signal_Start', 'Score', 'Met_Conditions']
        print(res_df_top[display_cols])
        
        # CSV保存
        res_df_all = pd.DataFrame(results)
        res_df_all.to_csv("analysis_results.csv", index=False)
        print(f"全 {len(results)} 件の結果を analysis_results.csv に保存しました。")
    else:
        print("条件を満たす銘柄はありませんでした。")

if __name__ == "__main__":
    main()
