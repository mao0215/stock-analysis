import os
import datetime
import time
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import io
import warnings

# FutureWarningなどを抑制
warnings.simplefilter(action='ignore', category=FutureWarning)

# 設定
DATA_DIR = "data"
JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
JPX_BASE_URL = "https://www.jpx.co.jp"
CHUNK_SIZE = 100  # 一括ダウンロードする銘柄数

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"ディレクトリを作成しました: {DATA_DIR}")

def get_domestic_tickers():
    """JPXウェブサイトから内国株式（プライム・スタンダード・グロース）の銘柄リストを取得します。"""
    print("JPXウェブサイトから銘柄リストを取得中...")
    try:
        res = requests.get(JPX_URL)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, "html.parser")
        
        excel_link = None
        for a in soup.find_all("a"):
            href = a.get("href")
            if href and "data_j.xls" in href:
                excel_link = href
                break
        
        if not excel_link:
            print("Excelファイルのリンクが見つかりませんでした。")
            return []

        if not excel_link.startswith("http"):
            excel_link = JPX_BASE_URL + excel_link
            
        print(f"Excelファイルをダウンロード中: {excel_link}")
        res_excel = requests.get(excel_link)
        res_excel.raise_for_status()
        
        df = pd.read_excel(io.BytesIO(res_excel.content))
        domestic_df = df[df['市場・商品区分'].astype(str).str.contains('内国株式')]
        tickers = domestic_df['コード'].astype(str) + ".T"
        
        print(f"内国株式銘柄数: {len(tickers)}")
        return tickers.tolist()

    except Exception as e:
        print(f"銘柄リスト取得エラー: {e}")
        return []

def get_last_date(file_path):
    """CSVを読み込み、最終日付をdatetime.dateオブジェクトとして返します。"""
    try:
        # 最後の数行だけ読むのが高速だが、pandasだと全読みになりがち
        # ここではシンプルにpandasで読む（ファイルサイズが小さいので許容範囲）
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        last_date_str = df.iloc[-1]['Date']
        return datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
    except Exception:
        return None

def group_tickers_by_start_date(tickers):
    """
    各銘柄の既存データを確認し、取得開始日ごとにグループ化します。
    戻り値: { start_date_str: [ticker1, ticker2, ...] }
    """
    print("既存データをスキャンしてグループ分けしています...")
    groups = {}
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365 * 3) # 3年前
    
    for ticker in tickers:
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        start_date = default_start
        
        if os.path.exists(file_path):
            last_date = get_last_date(file_path)
            if last_date:
                if last_date >= today:
                    # 最新なのでスキップ（リストには含めない）
                    continue
                # 翌日から取得
                start_date = last_date + datetime.timedelta(days=1)
        
        # 文字列キーにする
        s_date_str = start_date.strftime('%Y-%m-%d')
        if s_date_str not in groups:
            groups[s_date_str] = []
        groups[s_date_str].append(ticker)
        
    return groups

def save_data_chunk(data, tickers_in_chunk):
    """
    一括取得したデータを個別のCSVに保存/追記します。
    data: yfinanceから取得したDataFrame (MultiIndex columns)
    """
    # dataのindex（Date）を文字列に変換
    data.index = data.index.strftime('%Y-%m-%d')
    
    # yfinanceのバグや仕様で、1銘柄だけだとMultiIndexにならない場合がある
    # group_by='ticker' を指定していても、ダウンロードできたのが1つだけだと挙動が変わることがあるので注意
    # しかし chunk_size=100 なら基本MultiIndexになるはず
    
    # 取得できた銘柄を確認
    # columnsのレベル0がTicker
    if isinstance(data.columns, pd.MultiIndex):
        downloaded_tickers = data.columns.levels[0].unique()
    else:
        # 単一銘柄の場合や、フラットな場合（あまりないが）
        # ここではdownloaded_tickersを特定するのが難しいが、
        # yfinance 0.2系で group_by='ticker' なら Ticker がトップレベルになる
        downloaded_tickers = [tickers_in_chunk[0]] if len(tickers_in_chunk) == 1 else []

    for ticker in tickers_in_chunk:
        try:
            # 該当銘柄のデータを抽出
            if ticker not in data.columns:
                # 取得失敗、またはデータなし
                continue
                
            df_ticker = data[ticker].copy()
            
            # 全てNaNの行を削除
            df_ticker = df_ticker.dropna(how='all')
            
            if df_ticker.empty:
                continue
            
            # 必要なカラム: Open, High, Low, Close, Volume
            # yfinanceは大文字始まり
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            # 存在しないカラムがあればNaNで埋めるか、無視するか
            # 基本はあるはず
            
            # インデックスを列に戻す
            df_ticker = df_ticker.reset_index()
            df_ticker.rename(columns={'index': 'Date', 'Date': 'Date'}, inplace=True) # index名がDateの場合とそうでない場合
            
            # カラム確認
            available_cols = [c for c in required_cols if c in df_ticker.columns]
            if not available_cols:
                continue
                
            final_df = df_ticker[['Date'] + available_cols]
            
            file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
            
            # 追記か新規か
            if os.path.exists(file_path):
                # ヘッダーなしで追記
                final_df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                # 新規保存
                final_df.to_csv(file_path, index=False)
                
        except Exception as e:
            print(f"[{ticker}] 保存エラー: {e}")

def main():
    ensure_data_dir()
    
    all_tickers = get_domestic_tickers()
    if not all_tickers:
        print("銘柄リストが取得できませんでした。終了します。")
        return

    # グループ分け
    groups = group_tickers_by_start_date(all_tickers)
    
    total_download_count = sum(len(ts) for ts in groups.values())
    print(f"取得対象銘柄数: {total_download_count} / 全 {len(all_tickers)} 銘柄")
    
    if total_download_count == 0:
        print("すべてのデータが最新です。")
        return

    # グループごとに処理
    for start_date_str, tickers in groups.items():
        print(f"\n開始日 {start_date_str} のグループ ({len(tickers)} 銘柄) を処理中...")
        
        # チャンク分割
        for i in range(0, len(tickers), CHUNK_SIZE):
            chunk_tickers = tickers[i : i + CHUNK_SIZE]
            print(f"  - Chunk {i//CHUNK_SIZE + 1}: {len(chunk_tickers)} 銘柄を取得中...")
            
            try:
                # 一括ダウンロード
                # auto_adjust=False: Open/Closeをそのまま取得（調整後終値はAdj Closeに入るが今回は使わない）
                # threads=True: マルチスレッド有効
                data = yf.download(
                    chunk_tickers, 
                    start=start_date_str, 
                    group_by='ticker', 
                    auto_adjust=False, 
                    threads=True,
                    progress=False
                )
                
                if data.empty:
                    print("    データが見つかりませんでした。")
                    continue
                    
                # 保存
                save_data_chunk(data, chunk_tickers)
                
            except Exception as e:
                print(f"    ダウンロードエラー: {e}")
                
            # 負荷軽減のための短い待機（連続リクエスト対策）
            time.sleep(1)

    print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()
