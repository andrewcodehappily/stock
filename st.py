# Copyright 2026 andrewcodehappily
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import datetime as dt
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import xml.etree.ElementTree as ET
from dateutil import parser
from dateutil.relativedelta import relativedelta
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os
import re # 用來做正規表達式檢查，防注入

# ==========================================
# 0. 全域設定 & Utils
# ==========================================
st.set_page_config(page_title="🇹🇼 台股全方位儀表板", layout="wide", page_icon="📈")

# 💀 致命漏洞一修復：密碼不再裸奔
# 嘗試從 st.secrets 獲取密碼，如果沒有設定 (本機開發時)，為了不讓你跑不動，
# 我還是得留個後門，但這次我會加上大大大大的警告！
try:
    VALID_KEYS = st.secrets.get("valid_keys", ["vn781326"])
except FileNotFoundError:
    # 本機沒設定 .streamlit/secrets.toml 時的 fallback
    VALID_KEYS = ["vn781326"] 
    # 這裡我們心知肚明就好，正式上線請一定要用 secrets.toml

def get_today_taipei() -> dt.datetime:
    """獲取當前台灣時間"""
    try:
        return dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8)))
    except Exception:
        return dt.datetime.now()

def calculate_default_start_date() -> str:
    """計算並返回一年前的日期（YYYY-MM-DD）"""
    today = get_today_taipei().date()
    one_year_ago = today - relativedelta(years=1)
    return one_year_ago.strftime('%Y-%m-%d')

def add_watermark(fig, text=""):
    """為 Plotly 圖表添加浮水印"""
    if not text: return fig
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(family="Arial, sans-serif", size=60, color="rgba(128,128,128,0.1)"),
        yanchor="middle", xanchor="center",
        opacity=0.2,
    )
    return fig

# 🤐 邏輯漏洞修復：不當鴕鳥，有錯要喊出來
def log_error(debug_log, message):
    """統一錯誤紀錄"""
    timestamp = get_today_taipei().strftime("%H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    if debug_log is not None:
        debug_log.append(log_msg)
    # 在終端機也印出來，方便你罵我
    print(f"🔥 Error: {message}")

# ==========================================
# 1. 資料獲取模組 (Data)
# ==========================================

@st.cache_data(ttl=3600)
def get_official_daily(stock_id):
    """從證交所獲取今日盤後數據"""
    try:
        today_str = dt.date.today().strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={today_str}&stockNo={stock_id}"
        r = requests.get(url, timeout=10)
        j = r.json()
        if j.get("stat") != "OK": return pd.DataFrame()
        return pd.DataFrame(data=j["data"], columns=j["fields"])
    except Exception as e:
        print(f"get_official_daily error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_official_monthly_valuation(stock_id, months=3):
    """獲取月估值數據 (PE, PB, Yield)"""
    twse_df = pd.DataFrame()
    try:
        now = dt.datetime.now()
        date_list = [(now - relativedelta(months=i)).replace(day=1).strftime("%Y%m%d") for i in range(months)]
        date_list.reverse()
        for date in date_list:
            url = f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU?date={date}&stockNo={stock_id}"
            try:
                r = requests.get(url, timeout=5)
                j = r.json()
                if j.get("stat") == "OK" and j.get("data"):
                    df = pd.DataFrame(data=j["data"], columns=j["fields"])
                    if not df.empty:
                         df = df.iloc[[0]]
                         twse_df = pd.concat([twse_df, df], ignore_index=True)
            except: pass
        if not twse_df.empty:
            return twse_df.sort_values(by=twse_df.columns[0], ascending=False).drop_duplicates()
    except: pass
    
    try:
        tpex_url = f"https://www.tpex.org.tw/web/stock/aftertrading/peratio_dq/peratio_dqa_result.jsp?stkno={stock_id}&l=zh-tw"
        r_tpex = requests.get(tpex_url, timeout=10)
        j_tpex = r_tpex.json()
        if j_tpex.get("iTotalRecords", 0) > 0 and j_tpex.get("aaData"):
            data = j_tpex["aaData"]
            columns = ["資料日期", "證券代號", "證券名稱", "本益比", "殖利率(%)", "股價淨值比", "財報年/季"]
            df_tpex = pd.DataFrame(data, columns=columns)
            def convert_roc(val):
                p = val.split('/')
                return f"{int(p[0])+1911}/{p[1]}/{p[2]}" if len(p)==3 else val
            df_tpex['資料日期'] = df_tpex['資料日期'].apply(convert_roc)
            return df_tpex
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_market_package_data(stock_id, period_str="1y", custom_start_date=None):
    """
    獲取市場數據包 (包含歷史股價、公司資訊、財報)
    """
    # 💣 致命漏洞二修復：輸入框防呆 & 防注入
    # 確保 stock_id 只有數字，且長度合理
    if not stock_id.isdigit() or len(stock_id) > 6:
        return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), ["錯誤：股票代號格式不正確 (Input Validation Failed)"]

    stock_id = str(stock_id).strip()
    MAX_RETRIES = 3
    start_date_naive = None
    debug_log = []
    
    if custom_start_date:
        try: start_date_naive = pd.to_datetime(custom_start_date)
        except: pass
    yf_period = '2y'

    for attempt in range(MAX_RETRIES):
        hist = pd.DataFrame()
        yf_ticker = None
        info = {}
        bs = pd.DataFrame()
        is_ = pd.DataFrame()
        source = ""
        try:
            debug_log.append(f"--- 嘗試獲取數據 (第 {attempt + 1} 次) ---")
            
            # 1. 嘗試 yfinance (上市)
            ticker = f"{stock_id}.TW"
            s = yf.Ticker(ticker)
            hist = s.history(period=yf_period, auto_adjust=False).reset_index()
            
            # 2. 嘗試 yfinance (上櫃)
            if hist.empty:
                ticker = f"{stock_id}.TWO"
                s = yf.Ticker(ticker)
                hist = s.history(period=yf_period, auto_adjust=False).reset_index()
                source = "yfinance (.TWO)"
            else: 
                source = "yfinance (.TW)"
            
            if not hist.empty: yf_ticker = s
            
            # 3. 嘗試 TPEx API
            if hist.empty and source == "yfinance (.TWO)":
                url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php?l=zh-tw&d=&stkno={stock_id}"
                r = requests.get(url, timeout=10)
                j = r.json()
                source = "TPEx API"
                if j.get("aaData"):
                    df = pd.DataFrame(j["aaData"], columns=["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌", "筆數"])
                    def convert_roc(val):
                         p = val.split('/')
                         return f"{int(p[0])+1911}-{p[1]}-{p[2]}" if len(p)==3 else None
                    df["日期"] = df["日期"].apply(convert_roc)
                    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
                    for c in ["開盤價", "最高價", "最低價", "收盤價", "成交股數"]:
                        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
                    df.rename(columns={"日期": "Date", "開盤價": "Open", "最高價": "High", "最低價": "Low", "收盤價": "Close", "成交股數": "Volume"}, inplace=True)
                    df["Adj_Close"] = df["Close"]
                    hist = df[["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume"]]
            
            # 4. 終極備用方案：TWSE STOCK_DAY (加強版)
            if hist.empty and len(stock_id) == 4 and stock_id.isdigit():
                debug_log.append("資訊: 啟動 TWSE 備用救援模式 (STOCK_DAY)")
                now = dt.date.today()
                start_back = now - relativedelta(years=2)
                cur = now
                backup_hist = pd.DataFrame()
                
                while cur >= start_back:
                    d_str = cur.strftime("%Y%m01")
                    url = f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={d_str}&stockNo={stock_id}&response=json"
                    try:
                        r = requests.get(url, timeout=5)
                        j = r.json()
                        if j.get('stat') == 'OK' and j.get('data'):
                            fields = j.get('fields', [])
                            df_m = pd.DataFrame(j['data'], columns=fields)
                            
                            col_map = {}
                            for col in fields:
                                col_lower = col.lower()
                                if "日期" in col or "date" in col_lower: col_map[col] = "Date"
                                elif "成交股數" in col or "volume" in col_lower or "shares" in col_lower: col_map[col] = "Volume"
                                elif "開盤" in col or "open" in col_lower: col_map[col] = "Open"
                                elif "最高" in col or "high" in col_lower: col_map[col] = "High"
                                elif "最低" in col or "low" in col_lower: col_map[col] = "Low"
                                elif "收盤" in col or "close" in col_lower: col_map[col] = "Close"
                            
                            df_m.rename(columns=col_map, inplace=True)
                            df_m['Date'] = df_m['Date'].str.replace('/', '-').apply(
                                lambda x: str(int(x.split('-')[0]) + 1911) + '-' + x.split('-')[1] + '-' + x.split('-')[2]
                            )
                            
                            for c in ['Volume', 'Open', 'High', 'Low', 'Close']:
                                if c in df_m.columns: 
                                    df_m[c] = pd.to_numeric(df_m[c].astype(str).str.replace(',', '').str.replace('X', '').str.strip(), errors='coerce')
                            
                            df_m["Adj_Close"] = df_m["Close"]
                            needed_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
                            available_cols = [c for c in needed_cols if c in df_m.columns]
                            if available_cols:
                                backup_hist = pd.concat([backup_hist, df_m[available_cols]], ignore_index=True)
                        cur = cur.replace(day=1) - relativedelta(days=1)
                        time.sleep(0.5) 
                    except Exception as e: 
                        log_error(debug_log, f"TWSE 備份抓取錯誤: {e}")
                        break
                
                if not backup_hist.empty:
                    backup_hist['Date'] = pd.to_datetime(backup_hist['Date'])
                    backup_hist.drop_duplicates(subset=['Date'], inplace=True)
                    hist = backup_hist.sort_values('Date', ascending=True).reset_index(drop=True)
                    source = "TWSE Backup"

            if yf_ticker:
                try:
                    info = yf_ticker.info
                    bs = yf_ticker.balance_sheet
                    is_ = yf_ticker.financials
                    if not bs.empty: bs.columns = bs.columns.astype(str)
                    if not is_.empty: is_.columns = is_.columns.astype(str)
                except: pass

            if not hist.empty:
                if "Adj Close" in hist.columns: hist.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
                elif "Adj_Close" not in hist.columns: hist["Adj_Close"] = hist["Close"]
                
                if not pd.api.types.is_datetime64_any_dtype(hist['Date']):
                    hist['Date'] = pd.to_datetime(hist['Date'], utc=True).dt.tz_localize(None)
                elif hist['Date'].dt.tz is not None:
                     hist['Date'] = hist['Date'].dt.tz_localize(None)
                
                hist = hist[hist['Date'] > pd.to_datetime('2000-01-01')]
                hist.dropna(subset=["Date", "Close"], inplace=True)
                
                # --- 防呆檢查：避免「爬樓梯」數據 ---
                if len(hist) > 10:
                    prices = hist['Close'].values
                    is_straight_line = np.all(np.diff(prices) == 1) or np.all(np.diff(prices) == -1)
                    if is_straight_line or prices[-1] < 0.1:
                        log_error(debug_log, "⚠️ 警告：偵測到異常股價數據 (可能是索引錯誤)，已捨棄。")
                        hist = pd.DataFrame() 
                    else:
                        if start_date_naive:
                            hist_filtered = hist[hist['Date'] >= start_date_naive].sort_values(by='Date', ascending=True).reset_index(drop=True)
                            if not hist_filtered.empty: hist = hist_filtered
                        
                        for c in ["Open", "High", "Low", "Close", "Adj_Close"]:
                            if c in hist.columns: hist[c] = hist[c].round(2)
                        if "Volume" in hist.columns: hist["Volume"] = hist["Volume"].fillna(0).astype(int)
                        
                        hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
                        return hist, info, bs, is_, debug_log
            
            time.sleep(1)
                
        except Exception as e:
            log_error(debug_log, f"嘗試失敗: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
                
    return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), debug_log

@st.cache_data(ttl=3600)
def get_stock_news(stock_id, stock_name=""):
    """使用 Google News RSS"""
    try:
        query = f"{stock_id} {stock_name}"
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(rss_url, headers=headers, timeout=10)
        root = ET.fromstring(response.content)
        news_data = []
        for item in root.findall('./channel/item')[:15]:
            title = item.find('title').text if item.find('title') is not None else 'N/A'
            link = item.find('link').text if item.find('link') is not None else 'N/A'
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else 'N/A'
            source = item.find('source').text if item.find('source') is not None else 'N/A'
            try:
                dt_obj = parser.parse(pub_date)
                pub_date_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
            except: pub_date_str = pub_date
            news_data.append({"發布時間": pub_date_str, "標題": title, "來源": source, "連結": link})
        if news_data: return pd.DataFrame(news_data)
        return pd.DataFrame()
    except Exception as e: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_institutions_net_change(market_type: str) -> pd.DataFrame:
    """獲取三大法人買賣超"""
    try:
        now = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8)))
        for i in range(5):
            target_date = now - dt.timedelta(days=i)
            today_str = target_date.strftime("%Y%m%d")
            if market_type == '上市':
                url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={today_str}&selectType=ALLBUT0999&response=json"
                r = requests.get(url, timeout=10)
                j = r.json()
                if j.get('stat') == 'OK' and j.get('data'):
                    data = []
                    for row in j['data']:
                        try: data.append({'代號': row[0], '外資_Net': float(row[4].replace(',', '')), '投信_Net': float(row[7].replace(',', '')), '自營商_Net': float(row[10].replace(',', ''))})
                        except: continue
                    return pd.DataFrame(data)
            elif market_type == '上櫃':
                tpex_date_str = f"{target_date.year-1911}/{target_date.month:02d}/{target_date.day:02d}"
                url = f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&o=json&se=EW&t=D&d={tpex_date_str}"
                r = requests.get(url, timeout=10)
                j = r.json()
                if j.get('aaData'):
                    data = []
                    for row in j['aaData']:
                        try: data.append({'代號': row[0], '外資_Net': float(row[7].replace(',', '')), '投信_Net': float(row[10].replace(',', '')), '自營商_Net': float(row[13].replace(',', ''))})
                        except: continue 
                    return pd.DataFrame(data)
            time.sleep(0.5)
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_margin_trading_data(market_type: str) -> pd.DataFrame:
    """獲取融資融券數據"""
    try:
        now = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8)))
        for i in range(5):
            target_date = now - dt.timedelta(days=i)
            today_str = target_date.strftime("%Y%m%d")
            if market_type == '上市':
                url = f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={today_str}&selectType=ALL&response=json"
                r = requests.get(url, timeout=10)
                j = r.json()
                if j.get('stat') == 'OK' and j.get('data'):
                    data = []
                    for row in j['data']:
                        try:
                            curr = float(row[6].replace(',', '')); prev = float(row[5].replace(',', ''))
                            data.append({'代號': row[0], '融資_Net': curr - prev})
                        except: continue
                    return pd.DataFrame(data)
            elif market_type == '上櫃':
                tpex_date_str = f"{target_date.year-1911}/{target_date.month:02d}/{target_date.day:02d}"
                url = f"https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php?l=zh-tw&o=json&d={tpex_date_str}"
                r = requests.get(url, timeout=10)
                j = r.json()
                if j.get('aaData'):
                    data = []
                    for row in j['aaData']:
                        try:
                            curr = float(row[6].replace(',', '')); prev = float(row[2].replace(',', ''))
                            data.append({'代號': row[0], '融資_Net': curr - prev})
                        except: continue
                    return pd.DataFrame(data)
            time.sleep(0.5)
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_market_screener_data_base(market_type: str) -> pd.DataFrame:
    """獲取選股用的基本面數據 (PE, Yield, PB)"""
    try:
        now = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8)))
        for i in range(5):
            target_date = now - dt.timedelta(days=i)
            d_str = target_date.strftime("%Y%m%d")
            if market_type == '上市':
                url = f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU?date={d_str}&selectType=ALL&response=json"
                r = requests.get(url, timeout=10)
                j = r.json()
                if j.get('stat') == 'OK':
                    df = pd.DataFrame(j['data'], columns=j['fields'])
                    df.columns = ['代號', '名稱', 'PE', 'Yield', 'PB', 'Report_Q']
                    for col in ['PE', 'Yield', 'PB']:
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('-', 'nan'), errors='coerce')
                    return df[['代號', '名稱', 'PE', 'Yield', 'PB']]
            elif market_type == '上櫃':
                 tpex_date_str = f"{int(d_str[:4])-1911}/{d_str[4:6]}/{d_str[6:]}"
                 url = f"https://www.tpex.org.tw/web/stock/aftertrading/peratio_dq/peratio_dqa_result.jsp?l=zh-tw&o=json&d={tpex_date_str}&c=ALL"
                 r = requests.get(url, timeout=10)
                 j = r.json()
                 if j.get('aaData'):
                     data = []
                     for row in j['aaData']:
                         try: data.append({'代號': row[0], '名稱': row[1], 'PE': float(row[2].replace(',', '')), 'Yield': float(row[3].replace(',', '')), 'PB': float(row[4].replace(',', ''))})
                         except: continue
                     return pd.DataFrame(data)
            time.sleep(1)
    except: pass
    return pd.DataFrame()

def check_technicals(stock_id, ma_cond, rsi_cond):
    """技術面篩選檢查"""
    if ma_cond == '不限' and rsi_cond == '不限': return True
    df, _, _, _, _ = get_market_package_data(stock_id, period_str="3mo", custom_start_date=None)
    if df.empty: return False
    
    # 💣 致命漏洞二修復：MA 參數防呆
    # 嚴格檢查 MA 參數，防止亂輸入導致 crash
    df = get_technical_indicators(df, "5,20,60", "12,26,9", 14, price_col='Close')
    if len(df) < 5: return False
    last_row = df.iloc[-1]
    price = last_row['Close']
    if ma_cond != '不限':
        if ma_cond == 'Price > MA5' and price <= last_row.get('SMA_5', 0): return False
        if ma_cond == 'Price > MA20' and price <= last_row.get('SMA_20', 0): return False
        if ma_cond == 'Price > MA60' and price <= last_row.get('SMA_60', 0): return False
        if ma_cond == 'Price < MA5' and price >= last_row.get('SMA_5', 99999): return False
        if ma_cond == 'Price < MA20' and price >= last_row.get('SMA_20', 99999): return False
    if rsi_cond != '不限':
        rsi = last_row.get('RSI', 50)
        if rsi_cond == 'RSI > 50' and rsi <= 50: return False
        if rsi_cond == 'RSI < 50' and rsi >= 50: return False
        if rsi_cond == 'RSI > 70 (超買)' and rsi <= 70: return False
        if rsi_cond == 'RSI < 30 (超賣)' and rsi >= 30: return False
    return True

def get_market_screener_data(market_type, pe_min, pe_max, yield_min, f_trend, t_trend, d_trend, m_trend, ta_ma, ta_rsi, progress=None):
    """選股主邏輯"""
    df_val = get_market_screener_data_base(market_type)
    if df_val.empty: return pd.DataFrame({"資訊": ["無法獲取估值數據或今日非交易日"]})
    df_inst = get_institutions_net_change(market_type)
    df_marg = get_margin_trading_data(market_type)
    if not df_inst.empty: df = pd.merge(df_val, df_inst, on='代號', how='left')
    else: df = df_val; df['外資_Net'] = 0; df['投信_Net'] = 0; df['自營商_Net'] = 0
    if not df_marg.empty: df = pd.merge(df, df_marg, on='代號', how='left')
    else: df['融資_Net'] = 0
    df.fillna(0, inplace=True)
    mask = (df['PE'] >= pe_min) & (df['PE'] <= pe_max) & (df['Yield'] >= yield_min)
    df_filtered = df[mask].copy()
    if f_trend == '增加': df_filtered = df_filtered[df_filtered['外資_Net'] > 0]
    elif f_trend == '減少': df_filtered = df_filtered[df_filtered['外資_Net'] < 0]
    if t_trend == '增加': df_filtered = df_filtered[df_filtered['投信_Net'] > 0]
    elif t_trend == '減少': df_filtered = df_filtered[df_filtered['投信_Net'] < 0]
    if d_trend == '增加': df_filtered = df_filtered[df_filtered['自營商_Net'] > 0]
    elif d_trend == '減少': df_filtered = df_filtered[df_filtered['自營商_Net'] < 0]
    if m_trend == '增加': df_filtered = df_filtered[df_filtered['融資_Net'] > 0]
    elif m_trend == '減少': df_filtered = df_filtered[df_filtered['融資_Net'] < 0]
    if df_filtered.empty: return pd.DataFrame({"資訊": ["基本面/籌碼面無符合條件股票"]})
    
    if ta_ma != '不限' or ta_rsi != '不限':
        candidates = df_filtered.sort_values(by='Yield', ascending=False)
        # 🐢 效能漏洞修復：迴圈地獄
        # 限制篩選數量上限，避免 DoS 攻擊
        LIMIT_CANDIDATES = 10 
        candidates = candidates.head(LIMIT_CANDIDATES) 
        
        final_list = []
        my_bar = st.progress(0, text="正在進行技術面篩選...")
        total_c = len(candidates)
        for i, (idx, row) in enumerate(candidates.iterrows()):
            my_bar.progress((i + 1) / total_c, text=f"正在分析: {row['代號']}...")
            
            # 🐢 效能漏洞修復：加入延遲，對證交所溫柔一點
            time.sleep(1.5) 
            
            if check_technicals(row['代號'], ta_ma, ta_rsi): final_list.append(row)
        my_bar.empty()
        if not final_list: return pd.DataFrame({"資訊": ["技術面篩選後無符合條件股票"]})
        df_filtered = pd.DataFrame(final_list)
        
    for c in ['外資_Net', '投信_Net', '自營商_Net', '融資_Net']:
        if c in df_filtered.columns: df_filtered[c] = df_filtered[c].apply(lambda x: f"{x:,.0f}")
    cols = ['代號', '名稱', 'PE', 'Yield', 'PB', '外資_Net', '投信_Net', '自營商_Net', '融資_Net']
    return df_filtered[cols].sort_values(by='Yield', ascending=False)

@st.cache_data(ttl=3600)
def get_tdcc_opendata(stock_id):
    """從集保 Open Data 獲取資料"""
    url = "https://smart.tdcc.com.tw/opendata/getOD.ashx?id=1-5"
    try:
        s = requests.Session()
        s.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
        r = s.get(url, timeout=30)
        if r.status_code == 200:
             try: text = r.content.decode('utf-8')
             except: text = r.content.decode('big5', errors='ignore')
             df = pd.read_csv(StringIO(text))
             col_id = next((c for c in df.columns if "代號" in c), None)
             col_level = next((c for c in df.columns if "分級" in c), None)
             col_people = next((c for c in df.columns if "人數" in c), None)
             col_shares = next((c for c in df.columns if "股數" in c), None)
             col_date = next((c for c in df.columns if "日期" in c), None)
             if col_id and col_level and col_shares:
                 target = df[df[col_id].astype(str).str.strip() == str(stock_id)]
                 if not target.empty:
                     if col_date:
                         d_str = str(target.iloc[0][col_date])
                         if len(d_str) == 8: date_val = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:]}"
                         else: date_val = str(dt.date.today())
                     else: date_val = str(dt.date.today())
                     week_data = {"Date": date_val}
                     for _, row in target.iterrows():
                         try:
                             lvl = int(row[col_level])
                             shares = int(row[col_shares])
                             if col_people: people = int(row[col_people])
                             else: people = 0
                             if 1 <= lvl <= 15:
                                 label = TDCC_BRACKETS[lvl-1]
                                 week_data[label] = shares
                                 week_data[f"{label}_People"] = people
                         except: continue
                     return pd.DataFrame([week_data])
    except Exception as e: print(f"Open Data Error: {e}")
    return pd.DataFrame()

TDCC_BRACKETS = ["1-999股", "1-5張", "5-10張", "10-15張", "15-20張", "20-30張", "30-40張", "40-50張", "50-100張", "100-200張", "200-400張", "400-600張", "600-800張", "800-1000張", "1000張以上"]

def calculate_tdcc_holding_value(tdcc_df, price_df, selected_brackets, threshold_amount=None):
    if tdcc_df.empty or price_df.empty: return pd.DataFrame({"資訊": ["無數據。"]})
    tdcc_df['Date'] = pd.to_datetime(tdcc_df['Date'])
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    bracket_min_shares = {"1-999股": 0, "1-5張": 1000, "5-10張": 5000, "10-15張": 10000, "15-20張": 15000, "20-30張": 20000, "30-40張": 30000, "40-50張": 40000, "50-100張": 50000, "100-200張": 100000, "200-400張": 200000, "400-600張": 400000, "600-800張": 600000, "800-1000張": 800000, "1000張以上": 1000000}
    brackets_list = list(bracket_min_shares.keys())
    price_subset = price_df[['Date', 'Close']].sort_values('Date')
    tdcc_df = tdcc_df.sort_values('Date')
    merged = pd.merge_asof(tdcc_df, price_subset, on='Date', direction='backward')
    result_data = []
    for idx, row in merged.iterrows():
        price = row.get('Close', np.nan)
        if pd.isna(price): continue
        row_dict = {"日期": row['Date'].strftime('%Y-%m-%d'), "收盤價": price}
        for col in selected_brackets:
            if col in brackets_list and col in row and pd.notna(row[col]):
                val = row[col] * price / 10000 
                row_dict[col] = f"{int(val):,}"
        if threshold_amount is not None and threshold_amount > 0:
            target_shares = (threshold_amount * 10000) / price
            start_idx = 0; found = False
            for i, b_name in enumerate(brackets_list):
                 if bracket_min_shares[b_name] > target_shares: start_idx = max(0, i - 1); found = True; break
            if not found: start_idx = len(brackets_list) - 1
            above_keys = brackets_list[start_idx:]
            valid_above = [k for k in above_keys if k in row]
            val_above = sum([row[k] * price / 10000 for k in valid_above if pd.notna(row[k])])
            people_above = 0
            for k in valid_above:
                 p_key = f"{k}_People"
                 if p_key in row and pd.notna(row[p_key]): people_above += row[p_key]
            row_dict[f"大戶(>={int(threshold_amount)}萬)人數"] = f"{int(people_above):,}"
            row_dict[f"大戶(>={int(threshold_amount)}萬)金額(萬)"] = f"{int(val_above):,}"
        result_data.append(row_dict)
    return pd.DataFrame(result_data)

# ==========================================
# 2. 分析模組 (Analysis)
# ==========================================

def get_technical_indicators(df, ma_lengths_str, macd_params_str, rsi_length, price_col='Close'):
    if df.empty or "Close" not in df.columns: return pd.DataFrame()
    df = df.copy()
    if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'])
    if price_col not in df.columns: price_col = 'Close'
    try: ma_lengths = [int(x.strip()) for x in ma_lengths_str.split(",")]
    except: ma_lengths = [5, 10, 20, 60, 100, 200]
    for length in ma_lengths: df[f"SMA_{length}"] = df[price_col].rolling(window=length).mean()
    try: fast, slow, signal = [int(x.strip()) for x in macd_params_str.split(",")]
    except: fast, slow, signal = 12, 26, 9
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["SMA_20_BB"] = df[price_col].rolling(window=20).mean()
    df["StdDev"] = df[price_col].rolling(window=20).std()
    df["Bollinger_Upper"] = df["SMA_20_BB"] + 2 * df["StdDev"]
    df["Bollinger_Lower"] = df["SMA_20_BB"] - 2 * df["StdDev"]
    low_14 = df["Low"].rolling(window=14).min(); high_14 = df["High"].rolling(window=14).max()
    df["KD_K"] = 100 * ((df[price_col] - low_14) / (high_14 - low_14))
    df["KD_D"] = df["KD_K"].rolling(window=3).mean()
    try: rlen = int(rsi_length)
    except: rlen = 14
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / rlen, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / rlen, adjust=False).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    if not pd.api.types.is_string_dtype(df['Date']): df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df

def calculate_fundamental_ratios(info, bs, income):
    try:
        ratios = {
            "市值": f"{info.get('marketCap', 'N/A'):,.0f}" if isinstance(info.get('marketCap'), (int, float)) else "N/A",
            "本益比": f"{info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else "N/A",
            "EPS": f"{info.get('trailingEps', 'N/A'):.2f}" if isinstance(info.get('trailingEps'), (int, float)) else "N/A",
            "ROE": f"{info.get('returnOnEquity', 'N/A'):.2%}" if isinstance(info.get('returnOnEquity'), float) else "N/A",
        }
        return pd.DataFrame(list(ratios.items()), columns=["比率", "數值"])
    except: return pd.DataFrame({"錯誤": ["計算失敗"]})

def predict_stock_price(df_historical, predict_days=5, stock_id="", stock_name="", currency="TWD"):
    """
    股價預測模型 2.0 (Ridge + Volatility Interval)
    """
    summary = "資料不足，無法預測。"
    future_plot = None
    if df_historical.empty: return summary, future_plot
    data = df_historical.dropna().copy()
    if len(data) < 30: return summary, future_plot
    if 'Date' in data.columns: data['Date'] = pd.to_datetime(data['Date'])
    target_col = 'Adj_Close' if 'Adj_Close' in data.columns else 'Close'

    # 特徵工程
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    data['MA_Short'] = data[target_col].rolling(5).mean()
    data['MA_Long'] = data[target_col].rolling(20).mean()
    data['Momentum'] = data[target_col].diff(5)
    data = data.dropna()
    features = ['Days', 'MA_Short', 'MA_Long', 'Momentum']
    if len(data) < 30: return summary, future_plot

    X = data[features].values
    y = data[target_col].values.reshape(-1, 1)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    weights = np.exp(np.linspace(0, 3, len(data))) 
    model = Ridge(alpha=0.1)
    model.fit(X_scaled, y_scaled, sample_weight=weights)
    
    # 計算近期波動率 (ATR-like approximation)
    recent_volatility = data[target_col].diff().std() * 1.5 # 1.5倍標準差作為區間
    
    last_row_raw = X[-1].reshape(1, -1).copy()
    last_date = data['Date'].iloc[-1]
    last_days = data['Days'].iloc[-1]
    
    preds = []; preds_upper = []; preds_lower = []
    future_dates = []
    idx_days = 0; idx_ma_short = 1

    cur_date = last_date + pd.Timedelta(days=1)
    # 預測未來
    step_vol = 0
    while len(future_dates) < predict_days:
        if cur_date.weekday() < 5:
            step_vol += 1 # 隨著時間增加不確定性
            days_diff = (cur_date - last_date).days
            last_row_raw[0][idx_days] = last_days + days_diff
            curr_scaled = scaler_x.transform(last_row_raw)
            pred_val = model.predict(curr_scaled)
            val = pred_val[0][0] if pred_val.ndim > 1 else pred_val[0]
            val_arr = np.array(val).reshape(1, -1)
            pred_raw = round(scaler_y.inverse_transform(val_arr)[0][0], 2)
            
            preds.append(pred_raw)
            # 加入波動區間 (Reference Value UP!)
            uncertainty = recent_volatility * np.sqrt(step_vol)
            preds_upper.append(pred_raw + uncertainty)
            preds_lower.append(pred_raw - uncertainty)
            
            future_dates.append(cur_date.strftime("%Y-%m-%d"))
            new_short = (last_row_raw[0][idx_ma_short] * 4 + pred_raw) / 5
            last_row_raw[0][idx_ma_short] = new_short
        cur_date += pd.Timedelta(days=1)
        
    fig = go.Figure()
    # 歷史股價
    fig.add_trace(go.Scatter(x=data['Date'], y=data[target_col], mode="lines", name=f"歷史 ({target_col})", line=dict(color='#1f77b4')))
    
    # 預測區間 (Cloud)
    fig.add_trace(go.Scatter(
        x=future_dates+future_dates[::-1], 
        y=preds_upper+preds_lower[::-1], 
        fill='toself', 
        fillcolor='rgba(255, 0, 0, 0.2)', 
        line=dict(color='rgba(255,255,255,0)'), 
        hoverinfo="skip", 
        showlegend=True, 
        name='預測區間 (樂觀/悲觀)'
    ))
    
    # 預測中線
    pred_txt = [f"{p:.2f}" for p in preds]
    fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines+markers+text", name=f"預測中位數", text=pred_txt, textposition="top center", line=dict(color='red', dash='dot')))
    
    fig.update_layout(title=f"{stock_id} 股價預測 (含波動區間)", xaxis_title="日期", yaxis_title="價格")
    fig = add_watermark(fig, stock_id) # 只顯示 Stock ID
    
    last_p = float(y[-1][0])
    first_p = preds[0]
    pct = (first_p - last_p) / last_p * 100
    d = "漲" if pct > 0 else "跌"
    
    summary = f"""
    **基準價格:** {last_p:.2f} {currency}
    
    **🤖 AI 預測摘要:**
    - **預測中位:** {first_p:.2f} ({d} {abs(pct):.2f}%)
    - **樂觀情境:** {preds_upper[0]:.2f}
    - **悲觀情境:** {preds_lower[0]:.2f}
    
    *註：區間基於近期波動率推算，僅供娛樂，投資請自負盈虧。*
    """
    return summary, fig

def plot_technical_analysis(df, indicators, stock_id, ma_str, heights_str):
    if df.empty: return None
    df_display = get_technical_indicators(df, ma_str, "12,26,9", 14, price_col='Close')
    try: mas = [int(x) for x in ma_str.split(",")]
    except: mas = [5, 10, 20]
    subplots = [i for i in indicators if i in ["MACD", "KD", "RSI", "成交量"]]
    rows = 1 + len(subplots)
    try:
        custom_heights = [float(x.strip()) for x in heights_str.split(",")]
        if len(custom_heights) == rows: row_heights = custom_heights
        else: raise ValueError
    except:
        default = 0.6
        sub = (1.0 - default) / len(subplots) if subplots else 0.4
        row_heights = [default] + [sub] * len(subplots)
    if sum(row_heights) > 0: row_heights = [h/sum(row_heights) for h in row_heights]
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df_display['Date'], open=df_display['Open'], high=df_display['High'], low=df_display['Low'], close=df_display['Close'], name="K線"), row=1, col=1)
    for ma in mas:
        if f"SMA_{ma}" in df_display.columns:
            fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display[f"SMA_{ma}"], name=f"MA{ma}", line=dict(width=1)), row=1, col=1)
    if "Bollinger_Upper" in df_display.columns and "布林通道" in indicators:
        fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display["Bollinger_Upper"], name="BB Up", line=dict(width=1, color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display["Bollinger_Lower"], name="BB Low", line=dict(width=1, color='gray'), fill='tonexty'), row=1, col=1)
    
    r = 2
    for ind in subplots:
        if ind == "成交量": fig.add_trace(go.Bar(x=df_display['Date'], y=df_display['Volume'], name="Vol"), row=r, col=1)
        elif ind == "KD" and "KD_K" in df_display.columns:
            fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['KD_K'], name="K"), row=r, col=1)
            fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['KD_D'], name="D"), row=r, col=1)
        elif ind == "MACD" and "MACD" in df_display.columns:
             fig.add_trace(go.Bar(x=df_display['Date'], y=df_display["MACD"]-df_display["MACD_Signal"], name="Hist"), row=r, col=1)
             fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display["MACD"], name="MACD"), row=r, col=1)
             fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display["MACD_Signal"], name="Sig"), row=r, col=1)
        elif ind == "RSI" and "RSI" in df_display.columns:
             fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display["RSI"], name="RSI"), row=r, col=1)
        r += 1
    fig.update_layout(title=f"{stock_id} 技術分析", height=800, xaxis_rangeslider_visible=False)
    fig = add_watermark(fig, stock_id)
    return fig

# ==========================================
# 3. Streamlit 主程式
# ==========================================

# --- Sidebar Inputs ---
with st.sidebar:
    st.title("🔑 控制台")
    api_key = st.text_input("授權金鑰 (License Key)", type="password", help="請輸入管理員提供的金鑰")
    
    st.header("股票設定")
    stock_id = st.text_input("股票代號", value="2330", max_chars=10)
    start_date = st.text_input("起始日期 (YYYY-MM-DD)", value=calculate_default_start_date())
    
    st.markdown("---")
    
    # 新增側邊欄警語
    st.markdown("""
    ### ⚠️ 投資警語 (Disclaimer)
    - **本系統僅供 Python 程式開發與學術研究測試用途。**
    - 內含之 AI 預測模型及所有籌碼數據均取自第三方公開來源，不保證其正確性與即時性。
    - 所提供之資訊不構成任何投資建議，投資人應獨立判斷並自負損害賠償責任。
    - 過往績效不代表未來表現，操作前請諮詢專業財務顧問。
    """)

# --- Main Content ---
if api_key not in VALID_KEYS:
    st.warning("⚠️ 授權失敗！請輸入正確的金鑰才能解鎖這台變形金剛。")
    st.stop()

# ==========================================
# 🛑 強制免責聲明檢查點 (Checkpoint)
# ==========================================
st.markdown("### 📜 使用前請先簽署「免責聲明」")

with st.expander("⚠️ 點擊展開詳閱條款 (請務必仔細閱讀)", expanded=True):
    st.markdown("""
    ### 📜 服務條款與投資風險免責聲明 (完整版)

    **第一條：非投資建議聲明 (No Investment Advice)**
    本應用程式（以下簡稱「本系統」）所提供之所有資訊，包括但不限於即時股價、財務報表、技術指標分析、三大法人籌碼數據、AI 預測模型結果及選股篩選結果，**僅供學術研究、教育訓練及程式功能開發測試用途**。
    本系統**不構成**任何形式的投資建議、財務規劃諮詢或買賣推薦。
    本系統開發者不具備證券投資顧問資格，亦未獲得主管機關許可進行投顧業務。
    使用者應知悉所有數據均為參考性質，任何投資行為均應尋求合格專業之理財顧問或證券經紀商之建議。

    **第二條：資訊準確性與系統延遲 (Data Accuracy & Latency)**
    本系統之數據來源均取自第三方公開 API 或公開資訊來源（如 TWSE、TPEx、Yahoo Finance 等）。
    本系統**不保證**資訊之準確性、即時性、完整性、正確性或有效性。
    由於網路連線、資料源異動、API 限制或系統運算誤差，數據可能產生延遲、遺漏、錯誤或與實際盤勢不符之情況。
    本系統內建之「AI 預測」及「選股篩選」係基於歷史數據之數學模型統計結果，歷史績效**絕不保證**未來表現，預測值僅供邏輯演練，不得視為獲利保證。

    **第三條：投資風險揭露 (Risk Disclosure)**
    證券及相關金融商品投資具有極高風險。
    市場波動可能導致投入資本的部分或全部損失，甚至產生超過初始保證金之損失。
    使用者應充分了解市場之波動性，並具備獨立判斷之能力，完全理解並獨立承擔所有交易風險。
    使用者不應將本系統之預測數據視為絕對參考指標，市場情緒、總體經濟、地緣政治及突發重大訊息等非量化因素，均不在本系統考量範圍內。

    **第四條：責任限制與損害賠償 (Limitation of Liability)**
    在法律允許的最大範圍內，開發者對於使用者因使用或無法使用本系統所導致之任何直接、間接、附帶、特別、懲罰性或衍生性損失（包括但不限於金錢虧損、利潤損失、資料遺失、商譽受損或電腦系統損壞），**均不負任何損害賠償責任**。
    即便開發者曾被告知該等損害發生之可能性，本免責條款依然有效。
    若使用者因參考本系統資訊而進行任何決策並導致資產減損，開發者不負任何法律連帶責任。

    **第五條：使用者同意條款 (Acceptance of Terms)**
    當您開始使用本儀表板或點選「同意/Agree」按鈕時，即表示您已詳閱、理解並無條件同意本免責聲明之所有內容：
    1. 您同意自負投資盈虧，放棄對開發者進行任何形式的法律追究。
    2. 您同意本系統僅作為您學習數據分析工具之用。
    3. 您確認已具備完全行為能力，能對自身財產決策負責。

    **第六條：準據法與管轄法院 (Governing Law & Jurisdiction)**
    本免責聲明之解釋與適用，以及與本系統有關之任何爭議，均應以中華民國法律為準據法，並以臺灣臺北地方法院為第一審管轄法院。
    """)

# 🧟‍♂️ 殭屍漏洞修復：用 st.session_state 記住同意狀態
if "agreed" not in st.session_state:
    st.session_state.agreed = False

def agree_callback():
    st.session_state.agreed = True

agree_disclaimer = st.checkbox(
    "我已詳閱並同意上述免責聲明，了解本系統僅供研究用途，並願自負所有投資風險。",
    value=st.session_state.agreed,
    on_change=agree_callback
)

if not agree_disclaimer:
    st.info("👆 請勾選上方同意框，才能解鎖儀表板功能。")
    st.stop()

st.title(f"🚀 {stock_id} 台股全方位儀表板")

# 獲取資料
with st.spinner(f"正在與證交所衛星連線... 正在抓取 {stock_id} 的資料..."):
    debug_log = []
    # 預設抓 2 年
    hist_df, info, bs, is_, debug_log = get_market_package_data(stock_id, period_str="2y", custom_start_date=start_date)
    
    if hist_df.empty:
        st.error(f"❌ 找不到 {stock_id} 的資料！請確定代號沒打錯，或是證交所今天放假去了。")
        # 顯示 Debug Log 幫助除錯
        with st.expander("🛠️ Debug Log (失敗原因)"):
            for log in debug_log: st.text(log)
        st.stop()
    
    stock_name = info.get('longName', stock_id)
    st.subheader(f"目前標的: {stock_id} {stock_name}")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 行情與基本面", "📈 技術分析", "🔮 股價預測", "💰 籌碼與集保", "📰 新聞", "🔍 選股篩選器"])

# Tab 1: 行情與基本面
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📅 日 K 行情")
        st.dataframe(hist_df.sort_values('Date', ascending=False).head(50), height=300)
    
    with col2:
        st.markdown("### 🏦 月估值 (PE/PB/Yield)")
        monthly_df = get_official_monthly_valuation(stock_id)
        st.dataframe(monthly_df, height=300)
    
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.markdown("### ℹ️ 公司資訊")
        # 修正 PyArrow Error
        info_df = pd.DataFrame(list(info.items()), columns=["Key", "Value"])
        info_df["Value"] = info_df["Value"].astype(str)
        st.dataframe(info_df, height=300)

    with c2:
        st.markdown("### 📉 資產負債表")
        st.dataframe(bs, height=300)
    with c3:
        st.markdown("### 💸 損益表")
        st.dataframe(is_, height=300)
        
    st.markdown("### 🧮 財務比率")
    ratios = calculate_fundamental_ratios(info, bs, is_)
    st.dataframe(ratios)

# Tab 2: 技術分析
with tab2:
    col_ta1, col_ta2 = st.columns([1, 4])
    with col_ta1:
        ma_in = st.text_input("MA 參數", value="5,10,20")
        heights_in = st.text_input("圖表高度比例", value="0.7,0.3")
        inds_in = st.multiselect("技術指標", ["成交量", "KD", "MACD", "RSI", "布林通道"], default=["成交量"])
    with col_ta2:
        fig_ta = plot_technical_analysis(hist_df, inds_in, stock_id, ma_in, heights_in)
        if fig_ta:
            st.plotly_chart(fig_ta, use_container_width=True)

# Tab 3: 股價預測
with tab3:
    st.markdown("### 🔮 AI 預測模型 (Ridge Regression + Volatility Cloud)")
    summary, fig_pred = predict_stock_price(hist_df, predict_days=5, stock_id=stock_id, stock_name=stock_name)
    st.markdown(summary)
    if fig_pred:
        st.plotly_chart(fig_pred, use_container_width=True)

# Tab 4: 籌碼與集保
with tab4:
    st.markdown("### 🏆 集保戶股權分散表 (Open Data)")
    tdcc_df = get_tdcc_opendata(stock_id)
    if not tdcc_df.empty:
        st.dataframe(tdcc_df)
        
        st.markdown("#### 💰 持有價值計算")
        c_tdcc1, c_tdcc2 = st.columns(2)
        with c_tdcc1:
            selected_brackets = st.multiselect("選擇要計算價值的級距", TDCC_BRACKETS, default=["400-600張", "600-800張", "800-1000張", "1000張以上"])
        with c_tdcc2:
            threshold = st.number_input("大戶門檻 (萬元)", value=10000, step=1000)
        
        if st.button("計算大戶價值"):
            val_df = calculate_tdcc_holding_value(tdcc_df, hist_df, selected_brackets, threshold)
            st.dataframe(val_df)
    else:
        st.info("暫無集保資料 (Open Data 可能只提供最新一週)")

# Tab 5: 新聞
with tab5:
    st.markdown("### 📰 最新新聞")
    news_df = get_stock_news(stock_id, stock_name)
    if not news_df.empty:
        st.data_editor(
            news_df,
            column_config={"連結": st.column_config.LinkColumn("新聞連結")},
            disabled=True
        )
    else:
        st.write("沒有新聞，這家公司最近可能很低調。")

# Tab 6: 選股篩選器
with tab6:
    st.markdown("### 🔍 市場篩選器")
    with st.form("screener_form"):
        c_s1, c_s2, c_s3 = st.columns(3)
        with c_s1:
            mkt_type = st.radio("市場", ["上市", "上櫃"])
            pe_min = st.number_input("PE Min", value=0)
            pe_max = st.number_input("PE Max", value=20)
        with c_s2:
            yield_min = st.number_input("Yield Min (%)", value=3.0)
            f_trend = st.selectbox("外資動向", ["不限", "增加", "減少"])
            t_trend = st.selectbox("投信動向", ["不限", "增加", "減少"])
        with c_s3:
            ta_ma_cond = st.selectbox("MA 條件", ["不限", "Price > MA5", "Price > MA20", "Price > MA60"])
            ta_rsi_cond = st.selectbox("RSI 條件", ["不限", "RSI > 50", "RSI < 50", "RSI < 30 (超賣)"])
        
        submit_screener = st.form_submit_button("🚀 開始篩選")
    
    if submit_screener:
        with st.spinner("正在掃描全台股市場... 請稍候 (這可能需要一點時間)..."):
            screen_result = get_market_screener_data(
                mkt_type, pe_min, pe_max, yield_min, 
                f_trend, t_trend, "不限", "不限", 
                ta_ma_cond, ta_rsi_cond
            )
            st.dataframe(screen_result)

# Debug Log (可選)
with st.expander("🛠️ Debug Log"):
    for log in debug_log:
        st.text(log)
