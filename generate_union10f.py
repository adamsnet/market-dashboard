"""
UNION10F 個股武裝停損頁 — 讀 watchlist.txt, 生成 union10f.html
2026-08-10 升級為最終規則堆疊 (union10f_ratchet_decomp.py + union10f_padaptive2.py 驗證):
  觸發: 60日高點10%內 大黑K/長上影, 門檻 min(1*ATR10, 5%*股價)
  掛線: 觸發日高點-k*ATR10
  ★ 每日棘輪: 武裝後每日 line = max(line, trigger_high - k*今日ATR10), 只升不降
  ★ 停滯收緊 stall10_k275: 收盤 10 日未創持有期新高 → k 由 3.0 收至 2.75
  收盤跌破防線出場
  回場A: 收復觸發高點 (突破)
  回場B: 連續20交易日未創新低 (落底假設低接), 防線=底部低點
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import finlab
from finlab import data

token = os.environ.get('FINLAB_TOKEN', '')
if token:
    finlab.login(token)
else:
    finlab.login()

DASH = os.path.dirname(os.path.abspath(__file__))
K_EXIT = 3.0
K_STALL = 2.75      # 停滯時收緊後的 k
STALL_N = 10        # 收盤幾日未創持有期新高算停滯
NH_WIN = 60
ZONE = 0.9
PCT_CAP = 0.05
BOTTOM_DAYS = 20
PLOT_YEARS = 1

with open(os.path.join(DASH, 'watchlist.txt'), encoding='utf-8') as f:
    WATCH = [x.strip().upper() for x in f if x.strip() and not x.startswith('#')]


def is_tw(sid):
    """4碼純數字 = 台股, 其餘視為美股代號"""
    return len(sid) == 4 and sid.isdigit()


TW_LIST = [s for s in WATCH if is_tw(s)]
US_LIST = [s for s in WATCH if not is_tw(s)]
print(f"watchlist: 台股{TW_LIST} 美股{US_LIST}")

SRC = {}
if TW_LIST:
    SRC['TW'] = {
        'o': data.get('etl:adj_open'), 'h': data.get('etl:adj_high'),
        'l': data.get('etl:adj_low'), 'c': data.get('etl:adj_close'),
        'ro': data.get('price:開盤價'), 'rh': data.get('price:最高價'),
        'rl': data.get('price:最低價'), 'rc': data.get('price:收盤價'),
    }
    sc = data.get('security_categories')
    NAME = dict(zip(sc['stock_id'], sc['name']))
else:
    NAME = {}

if US_LIST:
    SRC['US'] = {
        'o': data.get('us_price:adj_open'), 'h': data.get('us_price:adj_high'),
        'l': data.get('us_price:adj_low'), 'c': data.get('us_price:adj_close'),
        'ro': data.get('us_price:open'), 'rh': data.get('us_price:high'),
        'rl': data.get('us_price:low'), 'rc': data.get('us_price:close'),
    }
    try:
        prof = data.get('us_company_profile')
        sub = prof[prof['symbol'].isin(US_LIST)]
        NAME.update(dict(zip(sub['symbol'].astype(str), sub['company_name'].astype(str))))
    except Exception as e:
        print(f"  (美股公司名稱取得失敗: {e})")


def run_stock(df, entry_i):
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(10).mean().values
    hh60p = pd.Series(h).rolling(NH_WIN).max().shift(1).values
    zone = h >= ZONE * hh60p
    thr = np.minimum(1.0 * atr, PCT_CAP * c)
    black = (c < o) & ((o - c) >= thr)
    ss = ((h - np.maximum(o, c)) >= thr) & (c < (l + 0.5 * (h - l)))
    trig = zone & (black | ss)

    hold = np.zeros(n, dtype=bool)
    line_arr = np.full(n, np.nan)
    ghost_arr = np.full(n, np.nan)   # 無現行防線時, 以舊防線值灰線延伸 (純參考)
    last_line = np.nan
    ev = {'trig': [], 'exit': [], 'reA': [], 'reB': []}
    hold[entry_i] = True
    in_pos = True
    line = trig_high = np.nan
    run_min = np.nan
    no_low = 0
    peak = np.nan
    stall = 0
    for i in range(entry_i + 1, n):
        if in_pos:
            if np.isnan(peak) or c[i] >= peak:
                peak = c[i]
                stall = 0
            else:
                stall += 1
            k = K_STALL if stall >= STALL_N else K_EXIT
            if trig[i] and atr[i] > 0:
                cand = h[i] - k * atr[i]
                line = cand if np.isnan(line) else max(line, cand)
                trig_high = h[i] if np.isnan(trig_high) else max(trig_high, h[i])
                ev['trig'].append(i)
            # 每日棘輪: 武裝後防線隨 ATR 收縮上移, 只升不降
            if not np.isnan(line) and not np.isnan(trig_high) and atr[i] > 0:
                line = max(line, trig_high - k * atr[i])
            line_arr[i] = line
            if not np.isnan(line):
                last_line = line
            elif not np.isnan(last_line):
                ghost_arr[i] = last_line     # 在場內但未武裝 (如 reA 回場後)
            if not np.isnan(line) and c[i] < line:
                in_pos = False
                ev['exit'].append(i)
                if np.isnan(trig_high):
                    trig_high = peak
                run_min = l[i]
                no_low = 0
            else:
                hold[i] = True
        else:
            if not np.isnan(last_line):
                ghost_arr[i] = last_line     # 出場期間灰線延伸
            if np.isnan(run_min) or l[i] < run_min:
                run_min = l[i]
                no_low = 0
            else:
                no_low += 1
            if not np.isnan(trig_high) and c[i] > trig_high:
                in_pos = True
                line = trig_high = np.nan
                peak = c[i]
                stall = 0
                ev['reA'].append(i)
                hold[i] = True
            elif no_low >= BOTTOM_DAYS:
                in_pos = True
                line = run_min
                line_arr[i] = line
                trig_high = np.nan
                peak = c[i]
                stall = 0
                ev['reB'].append(i)
                hold[i] = True
    return hold, line_arr, ghost_arr, ev, trig_high, in_pos, run_min, no_low, stall, line


cut = pd.Timestamp.today() - pd.DateOffset(years=PLOT_YEARS)
warm = cut - pd.DateOffset(years=1)

stocks = []
for sid in WATCH:
    mkt = 'TW' if is_tw(sid) else 'US'
    src = SRC.get(mkt)
    if src is None or sid not in src['c'].columns:
        print(f"  {sid}: 無資料, 跳過")
        continue
    df = pd.DataFrame({'open': src['o'][sid], 'high': src['h'][sid],
                       'low': src['l'][sid], 'close': src['c'][sid]}).dropna()
    df = df[df.index >= warm]
    if len(df) < 80:
        print(f"  {sid}: 資料不足, 跳過")
        continue
    entry_i = int(np.argmax(df.index >= cut))
    hold, line_arr, ghost_arr, ev, trig_high, in_pos, run_min, no_low, stall, line_now = run_stock(df, entry_i)

    # 還原價 → 實際價 換算係數 (逐日, 除權息日會跳動)
    dfr = pd.DataFrame({'open': src['ro'][sid], 'high': src['rh'][sid],
                        'low': src['rl'][sid], 'close': src['rc'][sid]}).reindex(df.index).ffill()
    ratio = (dfr['close'] / df['close']).values
    r_last = float(ratio[-1])

    m = df.index >= cut
    dfx = dfr[m]  # 顯示用真實 OHLC
    off = int(np.argmax(m))
    line_disp = line_arr * ratio  # 防線換算成真實價
    ghost_disp = ghost_arr * ratio
    # 用當前 live 防線, 不撈歷史陣列 (否則回場後未武裝會顯示出場前的殘影值)
    cur_line = line_now * r_last if in_pos and not np.isnan(line_now) else None

    cur = '$' if mkt == 'US' else ''
    stall_txt = (f"｜⚠停滯{stall}日 k=2.75 收緊中" if stall >= STALL_N
                 else f"｜停滯 {stall}/{STALL_N}")
    if in_pos and cur_line:
        detail = f"當前防線 {cur}{cur_line:.1f}{stall_txt}"
        color = '#f59e0b' if stall >= STALL_N else '#22c55e'
    elif in_pos:
        detail = f"尚無防線（等下次觸發K武裝）{stall_txt}"
        color = '#94a3b8'
    else:
        detail = (f"防線已破｜落底計時 {no_low}/{BOTTOM_DAYS}（低點 {cur}{run_min * r_last:.1f}）"
                  f"｜突破參考 {cur}{trig_high * r_last:.1f}")
        color = '#f59e0b'

    stocks.append({
        'id': sid, 'name': NAME.get(sid, ''), 'mkt': mkt, 'cur': cur,
        'detail': detail, 'color': color,
        'asof': df.index[-1].strftime('%m/%d'),
        'close': float(dfr['close'].iloc[-1]),
        'dates': [d.strftime('%y/%m/%d') for d in dfx.index],
        'o': [round(float(x), 2) for x in dfx['open']],
        'h': [round(float(x), 2) for x in dfx['high']],
        'l': [round(float(x), 2) for x in dfx['low']],
        'c': [round(float(x), 2) for x in dfx['close']],
        'line': [None if np.isnan(x) else round(float(x), 2) for x in line_disp[off:]],
        'ghost': [None if np.isnan(x) else round(float(x), 2) for x in ghost_disp[off:]],
        'ev': {k: [i - off for i in v if i >= off] for k, v in ev.items()},
    })
    print(f"  {sid} {NAME.get(sid,'')}: {detail}")

now_str = datetime.now().strftime('%Y-%m-%d %H:%M')

charts = ""
for s in stocks:
    charts += f"""
<div class="chart-container">
  <h3><span style="color:#e2e8f0">{s['id']} {s['name']}</span>{'<span style="color:#64748b; font-size:0.9em"> US</span>' if s['mkt'] == 'US' else ''}｜收盤 {s['cur']}{s['close']:,.2f}（{s['asof']}）｜<span style="color:{s['color']}">{s['detail']}</span></h3>
  <canvas id="ch_{s['id']}"></canvas>
</div>"""

html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UNION10F 個股武裝停損</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,'Segoe UI',sans-serif; background:#0f172a; color:#e2e8f0; padding:16px; }}
.header {{ text-align:center; margin-bottom:16px; }}
.header h1 {{ font-size:1.4em; color:#f8fafc; }}
.header .update {{ color:#94a3b8; font-size:0.85em; margin-top:4px; }}
.header a {{ color:#3b82f6; font-size:0.85em; }}
.charts-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; }}
@media (max-width: 900px) {{ .charts-grid {{ grid-template-columns:1fr; }} }}
.chart-container {{ background:#1e293b; border-radius:12px; padding:12px; border:1px solid #334155; }}
.chart-container h3 {{ color:#94a3b8; font-size:0.82em; margin-bottom:8px; }}
.chart-container canvas {{ max-height:260px; }}
.note {{ color:#64748b; font-size:0.8em; line-height:1.7; background:#1e293b; border-radius:12px; padding:14px; border:1px solid #334155; }}
</style>
</head>
<body>
<div class="header">
  <h1>UNION10F 個股武裝停損</h1>
  <div class="update">更新時間: {now_str}</div>
</div>
<div class="charts-grid">{charts}
</div>
<div class="note">
▼觸發K（60日高10%內 大黑K/長上影，門檻 min(1ATR, 5%價)）｜橘線=防線（觸發高點−k×ATR10，<b>每日棘輪只升不降</b>）｜灰虛線=舊防線延伸（僅參考位置，非現行停損）｜✕收盤跌破防線
藍▲收復觸發高點｜綠▲落底確認（連續20日未創新低，防線改掛底部低點）
<b>k 值</b>：平常 3.0；收盤連續 10 日未創持有期新高（停滯）→ 收緊至 2.75。
2026-08-10 升級：加入每日棘輪與停滯收緊（1,354 個歷史事件 + 2025 驗證段四項指標全勝；2026/07 帳戶重播七月少虧 37%）。
防線僅為風控參考，進出場由你自行決定。換股票：編輯 repo 的 dashboard/watchlist.txt，隔日 22:00 生效。
</div>
<script>
const S = {json.dumps(stocks, ensure_ascii=False)};
const gridC = '#334155', tickC = '#64748b';
for (const s of S) {{
  const n = s.dates.length;
  const wick = [], body = [], colors = [];
  for (let i = 0; i < n; i++) {{
    wick.push([s.l[i], s.h[i]]);
    body.push([Math.min(s.o[i], s.c[i]), Math.max(s.o[i], s.c[i])]);
    colors.push(s.c[i] >= s.o[i] ? '#ef4444' : '#22c55e');
  }}
  const mark = (idxs, color, rot, yFn) => {{
    const arr = new Array(n).fill(null);
    idxs.forEach(i => arr[i] = yFn(i));
    return {{ type: 'line', label: '', data: arr, showLine: false,
      pointStyle: 'triangle', rotation: rot, pointRadius: 5, pointHoverRadius: 6,
      pointBackgroundColor: color, pointBorderColor: '#fff', pointBorderWidth: 0.5, order: 0 }};
  }};
  new Chart(document.getElementById('ch_' + s.id), {{
    data: {{
      labels: s.dates,
      datasets: [
        {{ type: 'bar', label: '影線', data: wick, backgroundColor: colors, barPercentage: 0.25, categoryPercentage: 1.0, order: 3 }},
        {{ type: 'bar', label: 'K棒', data: body, backgroundColor: colors, barPercentage: 0.85, categoryPercentage: 1.0, order: 2 }},
        {{ type: 'line', label: '武裝出場線', data: s.line, borderColor: '#f97316', borderWidth: 2,
           pointRadius: (ctx) => {{ const d = ctx.dataset.data, i = ctx.dataIndex;
             return d[i] != null && (i === 0 || d[i-1] == null) && (i === d.length - 1 || d[i+1] == null) ? 3 : 0; }},
           pointBackgroundColor: '#f97316', pointBorderColor: '#f97316',
           spanGaps: false, stepped: true, order: 1 }},
        {{ type: 'line', label: '舊防線延伸', data: s.ghost, borderColor: '#64748b', borderWidth: 1.5,
           borderDash: [5, 4], pointRadius: 0, spanGaps: false, stepped: true, order: 1 }},
        mark(s.ev.trig, '#f59e0b', 180, i => s.h[i] * 1.01),
        mark(s.ev.exit, '#ef4444', 45, i => s.c[i]),
        mark(s.ev.reA, '#3b82f6', 0, i => s.l[i] * 0.99),
        mark(s.ev.reB, '#22c55e', 0, i => s.l[i] * 0.99),
      ]
    }},
    options: {{
      responsive: true, animation: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{ legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: (ctx) => {{
          if (ctx.dataset.label === 'K棒') {{ const i = ctx.dataIndex;
            return `開${{s.o[i]}} 高${{s.h[i]}} 低${{s.l[i]}} 收${{s.c[i]}}`; }}
          if (ctx.dataset.label === '武裝出場線' && ctx.raw != null) return `出場線 ${{ctx.raw}}`;
          if (ctx.dataset.label === '舊防線延伸' && ctx.raw != null) return `舊防線(參考) ${{ctx.raw}}`;
          return null; }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: tickC, maxTicksLimit: 12, font: {{ size: 10 }} }}, grid: {{ display: false }} }},
        y: {{ ticks: {{ color: tickC, font: {{ size: 10 }} }}, grid: {{ color: gridC }} }},
      }}
    }}
  }});
}}
</script>
</body>
</html>"""

out = os.path.join(DASH, 'union10f.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"OK -> {out}")
