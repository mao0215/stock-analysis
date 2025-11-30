import subprocess
import sys
import time
import pandas as pd

def run_script(script_name):
    print(f"=== {script_name} を実行中... ===")
    start_time = time.time()
    
    # python script_name を実行
    # リアルタイム出力を表示したい場合は工夫が必要だが、ここではシンプルに完了を待つ
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    
    elapsed = time.time() - start_time
    print(f"=== {script_name} 完了 (所要時間: {elapsed:.1f}秒) ===\n")
    
    if result.returncode != 0:
        print(f"エラー: {script_name} が異常終了しました。")
        sys.exit(1)

def main():
    print("【全銘柄シグナル監視システム】を開始します。\n")
    
    # 1. データ取得
    run_script("fetch_prices.py")
    
    # 2. 分析実行
    run_script("analyze_stocks.py")
    
    # 3. 結果サマリー表示
    print("=== 監視レポート ===")
    try:
        df = pd.read_csv("analysis_results.csv")
        
        # Score 6 の銘柄数
        score6_count = len(df[df['Score'] == 6])
        print(f"Score 6 (満点) の銘柄数: {score6_count} 件")
        
        # 直近（今日・昨日）シグナルが出た銘柄
        # 今日の日付を取得
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        yesterday = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        recent_signals = df[df['Signal_Start'] >= yesterday]
        
        if not recent_signals.empty:
            print(f"\n★ 直近 ({yesterday} 以降) にシグナルが出た銘柄: {len(recent_signals)} 件")
            print(recent_signals[['Ticker', 'Price', 'Signal_Start', 'Score']].to_string(index=False))
        else:
            print(f"\n直近 ({yesterday} 以降) にシグナルが出た銘柄はありませんでした。")
            
    except Exception as e:
        print(f"結果の読み込みに失敗しました: {e}")

    print("\n全工程が完了しました。")

if __name__ == "__main__":
    main()
