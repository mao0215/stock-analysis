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

def calculate_rsi(series, period=14):
    """RSIを計算する（ベクトル化）"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(series, fast=12, slow=26, signal=9):
    """MACDを計算する（ベクトル化）"""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def simulate_trade(df, entry_idx, tp_pct=0.10, sl_pct=0.05, max_days=10):
    """
    個別のトレードシミュレーション（ループ処理だが、バックテスト時のみ実行）
    利確(TP)と損切(SL)のどちらに先にヒットするか判定
    """
    entry_price = df.iloc[entry_idx]['Close']
    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)
    
    # 未来のデータを取得
    future_df = df.iloc[entry_idx+1 : entry_idx+1+max_days]
    
    if future_df.empty:
        return 0.0 # データ不足
        
    for _, row in future_df.iterrows():
        # 高値がTPを超えたか？
        if row['High'] >= tp_price:
            # ただし、同じ日に安値がSLを割っている可能性もある（ヒゲ）
            # 厳密には分足が必要だが、ここでは「始値に近い方」や「保守的にSL優先」などで判定
            # 今回はシンプルに「LowがSL割ってたらSL優先」とする（保守的）
            if row['Low'] <= sl_price:
                return -sl_pct
            return tp_pct
            
        # 安値がSLを割ったか？
        if row['Low'] <= sl_price:
            return -sl_pct
            
    # 期間内にどちらもヒットしなかった場合、最終日の終値で決済
    exit_price = future_df.iloc[-1]['Close']
    return (exit_price / entry_price) - 1.0

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
    price_diff = df['Close'].diff()
    vol_up = df['Volume'].where(price_diff > 0)
    vol_down = df['Volume'].where(price_diff < 0)
    
    avg_vol_up = vol_up.rolling(window=50, min_periods=10).mean()
    avg_vol_down = vol_down.rolling(window=50, min_periods=10).mean()
    
    c5 = avg_vol_up > avg_vol_down
    
    # --- Cond 6: 週足分析 (Resample & Broadcast) ---
    weekly_df = df.set_index('Date').resample('W').agg({
        'Close': 'last', 'Volume': 'sum'
    })
    weekly_diff = weekly_df['Close'].diff()
    weekly_vol_avg = weekly_df['Volume'].rolling(window=26, min_periods=5).mean()
    
    is_high_vol = weekly_df['Volume'] > weekly_vol_avg
    is_up = weekly_diff > 0
    is_down = weekly_diff < 0
    
    count_up_high_vol = (is_high_vol & is_up).astype(int).rolling(window=26, min_periods=5).sum()
    count_down_high_vol = (is_high_vol & is_down).astype(int).rolling(window=26, min_periods=5).sum()
    
    weekly_c6 = count_up_high_vol >= count_down_high_vol
    c6 = weekly_c6.reindex(df.set_index('Date').index, method='ffill').reset_index(drop=True)
    
    # --- Cond 4: トレンド (Higher Highs/Lows) ---
    high_peaks_idx = argrelextrema(df['High'].values, np.greater_equal, order=window)[0]
    low_peaks_idx = argrelextrema(df['Low'].values, np.less_equal, order=window)[0]
    
    high_peaks = pd.Series(np.nan, index=df.index)
    high_peaks.iloc[high_peaks_idx] = df['High'].iloc[high_peaks_idx]
    
    low_peaks = pd.Series(np.nan, index=df.index)
    low_peaks.iloc[low_peaks_idx] = df['Low'].iloc[low_peaks_idx]
    
    confirmed_high_peaks = high_peaks.shift(window)
    confirmed_low_peaks = low_peaks.shift(window)
    
    valid_high_idxs = confirmed_high_peaks.dropna().index
    valid_low_idxs = confirmed_low_peaks.dropna().index
    
    if len(valid_high_idxs) > 1:
        v_highs = confirmed_high_peaks.dropna()
        high_trend_ok = v_highs > v_highs.shift(1)
        s_high_trend = pd.Series(False, index=df.index)
        s_high_trend.loc[high_trend_ok.index] = high_trend_ok
        c4_high = s_high_trend.replace(False, np.nan).ffill().fillna(False)
    else:
        c4_high = pd.Series(False, index=df.index)

    if len(valid_low_idxs) > 1:
        v_lows = confirmed_low_peaks.dropna()
        low_trend_ok = v_lows > v_lows.shift(1)
        s_low_trend = pd.Series(False, index=df.index)
        s_low_trend.loc[low_trend_ok.index] = low_trend_ok
        c4_low = s_low_trend.replace(False, np.nan).ffill().fillna(False)
    else:
        c4_low = pd.Series(False, index=df.index)
        
    c4 = c4_high & c4_low
    
    # --- 新指標 ---
    rsi = calculate_rsi(df['Close'])
    c7 = rsi < 70
    
    macd, signal = calculate_macd(df['Close'])
    c8 = (macd > signal) & (macd > 0)
    
    trading_value = df['Close'] * df['Volume']
    avg_trading_value = trading_value.rolling(window=5).mean()
    c9 = avg_trading_value >= 100_000_000
    
    # --- スコア計算 ---
    score_series = (c1.astype(int) + c2.astype(int) + c3.astype(int) + 
                    c4.astype(int) + c5.astype(int) + c6.fillna(False).astype(int) +
                    c7.astype(int) + c8.astype(int) + c9.astype(int))
    
    return {
        'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5, 'c6': c6,
        'c7': c7, 'c8': c8, 'c9': c9,
        'score': score_series
    }

def process_ticker(ticker, params):
    """1銘柄を処理する関数（ベクトル化版 + 出口戦略バックテスト）"""
    try:
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        df = load_data(file_path)
        
        if df is None or len(df) < 200:
            return None

        vec_res = calculate_vectorized_conditions(df, params)
        score_series = vec_res['score']
        
        # --- バックテスト集計 (出口戦略あり) ---
        # ループ処理になるが、対象となる「高スコアの日」は少ないので高速
        backtest_stats = {}
        
        # 設定: 利確+10%, 損切-5%, 保有10日
        TP = 0.10
        SL = 0.05
        HOLD_DAYS = 10
        
        for s in [7, 8, 9]:
            # そのスコアの日付インデックスを取得
            indices = np.where(score_series == s)[0]
            
            returns = []
            for idx in indices:
                # 最終日付近はシミュレーションできないのでスキップ
                if idx >= len(df) - 1:
                    continue
                    
                ret = simulate_trade(df, idx, tp_pct=TP, sl_pct=SL, max_days=HOLD_DAYS)
                returns.append(ret)
            
            returns = np.array(returns)
            if len(returns) > 0:
                backtest_stats[s] = {
                    'count': len(returns),
                    'win_count': (returns > 0).sum(),
                    'total_return': returns.sum()
                }
            else:
                backtest_stats[s] = {'count': 0, 'win_count': 0, 'total_return': 0.0}
        
        # --- 最新の状態 ---
        current_score = score_series.iloc[-1]
        
        result_data = None
        if current_score >= 7:
            current_price = df.iloc[-1]['Close']
            last_date = df.iloc[-1]['Date']
            date_str = last_date.strftime('%Y-%m-%d')
            
            mask_below = score_series < 7
            rev_mask = mask_below.iloc[::-1]
            if rev_mask.any():
                last_below_idx = rev_mask.idxmax()
                start_idx = last_below_idx + 1
                if start_idx >= len(df): start_idx = len(df) - 1
            else:
                start_idx = 0
            
            signal_start = df.iloc[start_idx]['Date']
            start_str = signal_start.strftime('%Y-%m-%d')
            
            met_conditions = []
            for k in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']:
                if vec_res[k].iloc[-1]:
                    met_conditions.append(k.replace('c', 'cond'))
            
            # 推奨売買価格の計算
            entry_price = current_price
            target_price = int(entry_price * 1.10)
            stop_loss_price = int(entry_price * 0.95)
            
            result_data = {
                'Ticker': ticker,
                'Price': current_price,
                'Date': date_str,
                'Signal_Start': start_str,
                'Score': current_score,
                'Target_Price': target_price,
                'Stop_Loss': stop_loss_price,
                'Met_Conditions': met_conditions
            }
            
        return {
            'result': result_data,
            'backtest': backtest_stats
        }
            
    except Exception:
        return None
    return None

def main():
    print("分析を開始します（出口戦略バックテスト版）...")
    print("設定: 利確+10%, 損切-5%, 保有期間10日")
    start_time = time.time()
    
    files = os.listdir(DATA_DIR)
    tickers = [f.replace(".csv", "") for f in files if f.endswith(".csv")]
    print(f"対象銘柄数: {len(tickers)}")
    
    best_params = {'trend_window': 15}
    results = []
    
    global_backtest = {
        7: {'total_trades': 0, 'wins': 0, 'total_return_sum': 0.0},
        8: {'total_trades': 0, 'wins': 0, 'total_return_sum': 0.0},
        9: {'total_trades': 0, 'wins': 0, 'total_return_sum': 0.0}
    }
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_ticker, ticker, best_params) for ticker in tickers]
        
        completed_count = 0
        total_count = len(tickers)
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                if res['result']:
                    results.append(res['result'])
                
                bt_stats = res['backtest']
                for s in [7, 8, 9]:
                    if s in bt_stats:
                        global_backtest[s]['total_trades'] += bt_stats[s]['count']
                        global_backtest[s]['wins'] += bt_stats[s]['win_count']
                        global_backtest[s]['total_return_sum'] += bt_stats[s]['total_return']
            
            completed_count += 1
            if completed_count % 1000 == 0:
                print(f"進捗: {completed_count}/{total_count}")

    results.sort(key=lambda x: x['Score'], reverse=True)
    
    # --- 結果出力 ---
    output_lines = []
    
    def log(text):
        print(text)
        output_lines.append(text)

    elapsed = time.time() - start_time
    log(f"\n分析完了 (所要時間: {elapsed:.1f}秒)")
    
    # --- バックテスト結果表示 & 保存 ---
    log("\n=== 出口戦略バックテスト結果 (利確+10%, 損切-5%, 10日) ===")
    
    for s in [7, 8, 9]:
        stats = global_backtest[s]
        log(f"--- Score {s} ---")
        if stats['total_trades'] > 0:
            win_rate = (stats['wins'] / stats['total_trades']) * 100
            avg_return = (stats['total_return_sum'] / stats['total_trades']) * 100
            log(f"総取引回数: {stats['total_trades']} 回")
            log(f"勝率 (プラス決済): {win_rate:.2f}%")
            log(f"平均リターン (1トレードあたり): {avg_return:.2f}%")
        else:
            log("該当する取引データがありませんでした。")
            
    log("==========================================\n")
    
    log(f"条件を7つ以上満たした銘柄: {len(results)} 件")
    
    if results:
        res_df_top = pd.DataFrame(results[:20])
        # 表示列を絞り込む
        display_cols = ['Ticker', 'Price', 'Target_Price', 'Stop_Loss', 'Score']
        log("\n=== 推奨売買価格 (Top 20) ===")
        log(res_df_top[display_cols].to_string(index=False))
        
        res_df_all = pd.DataFrame(results)
        res_df_all.to_csv("analysis_results.csv", index=False)
        log(f"\n全 {len(results)} 件の結果を analysis_results.csv に保存しました。")
    else:
        log("条件を満たす銘柄はありませんでした。")
        
    # レポートファイルへの保存 (UTF-8)
    with open("latest_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print("\n詳細レポートを latest_report.txt に保存しました。")

if __name__ == "__main__":
    main()
