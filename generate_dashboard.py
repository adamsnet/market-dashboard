"""
市場風險儀表板 — 生成靜態 HTML
每天跑一次，生成 index.html 供 GitHub Pages 部署
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd, numpy as np, json, warnings
from datetime import datetime
warnings.filterwarnings('ignore')
from finlab import data
import finlab
import os
token = os.environ.get('FINLAB_TOKEN', '')
if token:
    finlab.login(token)
else:
    finlab.login()  # 用 finlab 內建的登入方式

import requests

from leverage_utils import (
    calc_breadth, _expanding_pct, _load_rvol, _load_pcr,
    _load_tsmc_holder, _load_margin_dd60, _load_margin_bal_chg
)

print("Loading data...")

# ============================================================
# 1. 計算所有指標
# ============================================================
# 報酬指數 (用於計算波動率等)
benchmark = data.get('benchmark_return:發行量加權股價報酬指數')
if isinstance(benchmark, pd.DataFrame):
    benchmark = benchmark.iloc[:, 0]
twii_return = benchmark.copy()

# 真正的加權指數 (從 Yahoo Finance)
try:
    resp = requests.get(
        'https://query1.finance.yahoo.com/v8/finance/chart/%5ETWII',
        params={'range': '6mo', 'interval': '1d'},
        headers={'User-Agent': 'Mozilla/5.0'},
        verify=False, timeout=30
    )
    ydata = resp.json()['chart']['result'][0]
    twii_real = pd.Series(
        ydata['indicators']['quote'][0]['close'],
        index=pd.to_datetime(ydata['timestamp'], unit='s').normalize()
    ).dropna()
    print(f"  TAIEX from Yahoo: {twii_real.iloc[-1]:.0f}")
except Exception as e:
    print(f"  Yahoo failed: {e}, using return index")
    twii_real = None

# 用報酬指數做所有計算 (回撤/波動等的相對值不受影響)
twii = twii_return.copy()

abv_ma20 = calc_breadth(20, 5)
abv_ma60 = calc_breadth(60, 5)

rvol_series, rvol_pct = _load_rvol()
pcr_series, pcr_pct = _load_pcr()

# PCR 原始值 (當日, 不平滑)
pcr_raw_df = data.get('tw_option_put_call_ratio')
pcr_oi_raw = pcr_raw_df.set_index('date')['買賣權未平倉量比率%']
pcr_oi_raw.index = pd.to_datetime(pcr_oi_raw.index)
pcr_oi_raw = pcr_oi_raw.sort_index()
tsmc_chg, tsmc_pct = _load_tsmc_holder()
margin_dd60 = _load_margin_dd60()
margin_bal_chg = _load_margin_bal_chg()

twii_high = twii.expanding().max()
drawdown = (twii / twii_high - 1) * 100
mom5 = twii.pct_change(5) * 100

# 合併 (最近90天)
end_date = twii.index[-1]
start_date = end_date - pd.Timedelta(days=120)

df = pd.DataFrame({
    'twii': twii, 'drawdown': drawdown,
    'abv_ma20': abv_ma20, 'abv_ma60': abv_ma60,
    'rvol': rvol_series, 'rvol_pct': rvol_pct,
    'pcr': pcr_series, 'pcr_pct': pcr_pct,
    'pcr_raw': pcr_oi_raw,
    'mom5': mom5,
    'margin_dd60': margin_dd60,
    'margin_bal_chg': margin_bal_chg,
}).dropna(how='all')

# 台積 (週頻 → 日頻)
df['tsmc_chg'] = tsmc_chg.reindex(df.index, method='ffill')
df['tsmc_pct'] = tsmc_pct.reindex(df.index, method='ffill')

df = df[df.index >= start_date]

# 當前值
latest = df.dropna(subset=['twii']).iloc[-1]
latest_date = df.dropna(subset=['twii']).index[-1]

# 訊號判斷
a20 = latest.get('abv_ma20', 0)
a60 = latest.get('abv_ma60', 0)

# 風險訊號
risk_v1 = (latest.get('tsmc_pct', 0.5) <= 0.20) and (latest.get('pcr_pct', 0.5) <= 0.25)
risk_v2 = (latest.get('rvol_pct', 0.5) >= 0.75) and (latest.get('tsmc_pct', 0.5) <= 0.25) and (latest.get('pcr_pct', 0.5) <= 0.30)

# 抄底
in_crash = latest.get('drawdown', 0) <= -10
mdd60 = latest.get('margin_dd60', 0)
bottom_signals = {
    '融資DD<-20%(斷頭潮)': in_crash and mdd60 < -20,
    '融資DD<-15%': in_crash and mdd60 < -15,
    '融資DD<-15%+動能>0': in_crash and mdd60 < -15 and latest.get('mom5', 0) > 0,
    'rvol>P90': in_crash and latest.get('rvol_pct', 0) >= 0.90,
    'PCR>P75': in_crash and latest.get('pcr_pct', 0) >= 0.75,
}

# ============================================================
# 1b. 歷史相似狀態統計 (2020起)
# ============================================================
print("  calculating similar historical states...")

# 用全期間的資料
df_full = pd.DataFrame({
    'twii': twii, 'rvol_pct': rvol_pct, 'pcr_pct': pcr_pct,
}).dropna()
df_full['tsmc_pct'] = tsmc_pct.reindex(df_full.index, method='ffill')
df_full = df_full[df_full.index >= '2020-01-01'].dropna()

for fwd in [5, 10, 20]:
    df_full[f'fwd_{fwd}d'] = twii.pct_change(fwd).shift(-fwd).reindex(df_full.index) * 100

# 當前各因子狀態
cur_rvol_high = latest.get('rvol_pct', 0) >= 0.75
cur_tsmc_low = latest.get('tsmc_pct', 0.5) <= 0.25
cur_pcr_low = latest.get('pcr_pct', 0) <= 0.30

# 統計各組合
similar_stats = []

combos = [
    ('目前狀態 (完全匹配)', [
        ('rvol_pct', '>=', 0.75) if cur_rvol_high else ('rvol_pct', '<', 0.75),
        ('tsmc_pct', '<=', 0.25) if cur_tsmc_low else ('tsmc_pct', '>', 0.25),
        ('pcr_pct', '<=', 0.30) if cur_pcr_low else ('pcr_pct', '>', 0.30),
    ]),
]

# 每個因子都列出當前狀態的回測
all_factors = [
    ('波動率高' if cur_rvol_high else '波動率正常',
     'rvol_pct', '>=' if cur_rvol_high else '<', 0.75),
    ('台積大戶撤' if cur_tsmc_low else '台積大戶正常',
     'tsmc_pct', '<=' if cur_tsmc_low else '>', 0.25),
    ('PCR低(自滿)' if cur_pcr_low else f'PCR偏高(排名{int(latest.get("pcr_pct", 0)*100)}%)',
     'pcr_pct', '<=' if cur_pcr_low else '>=', 0.30 if cur_pcr_low else latest.get('pcr_pct', 0.5) * 0.9),
]

# 各單因子
for name, col, op, th in all_factors:
    combos.append((f'只看{name}', [(col, op, th)]))

# 所有雙因子組合
for i in range(len(all_factors)):
    for j in range(i+1, len(all_factors)):
        n1, c1, o1, t1 = all_factors[i]
        n2, c2, o2, t2 = all_factors[j]
        combos.append((f'{n1}+{n2}', [(c1, o1, t1), (c2, o2, t2)]))

# 三重
combos.append(('三重全符合', [(c, o, t) for _, c, o, t in all_factors]))

# 對照
combos.append(('2020年以來全部日期', []))

for combo_name, conditions in combos:
    mask = pd.Series(True, index=df_full.index)
    for col, op, th in conditions:
        if op == '>=': mask &= df_full[col] >= th
        elif op == '<=': mask &= df_full[col] <= th
        elif op == '>': mask &= df_full[col] > th
        elif op == '<': mask &= df_full[col] < th

    sub = df_full[mask].dropna(subset=['fwd_20d'])
    if len(sub) < 3:
        continue

    similar_stats.append({
        'name': combo_name,
        'n': len(sub),
        'f5': sub['fwd_5d'].mean(),
        'f10': sub['fwd_10d'].mean(),
        'f20': sub['fwd_20d'].mean(),
        'wr5': (sub['fwd_5d'] > 0).mean() * 100,
        'wr20': (sub['fwd_20d'] > 0).mean() * 100,
    })

# 生成 HTML 表格
similar_html = '<div class="card" style="margin-bottom: 12px;">\n'
similar_html += '  <h3>歷史相似狀態 (2020年以來) 大盤後續報酬</h3>\n'
similar_html += '  <table>\n'
similar_html += '    <tr><th style="text-align:left">條件</th><th>樣本數</th><th>5日報酬</th><th>10日報酬</th><th>20日報酬</th><th>20日勝率</th></tr>\n'
for s in similar_stats:
    color_20 = '#ef4444' if s['f20'] < -1 else '#f59e0b' if s['f20'] < 0 else '#22c55e'
    bold = ' style="font-weight:700; background:#1a2332;"' if s['name'].startswith('目前') else ''
    similar_html += f'    <tr{bold}>'
    similar_html += f'<td style="text-align:left">{s["name"]}</td>'
    similar_html += f'<td>{s["n"]}</td>'
    similar_html += f'<td style="color: {"#ef4444" if s["f5"] < 0 else "#22c55e"}">{s["f5"]:+.2f}%</td>'
    similar_html += f'<td style="color: {"#ef4444" if s["f10"] < 0 else "#22c55e"}">{s["f10"]:+.2f}%</td>'
    similar_html += f'<td style="color: {color_20}">{s["f20"]:+.2f}%</td>'
    similar_html += f'<td>{s["wr20"]:.0f}%</td>'
    similar_html += '</tr>\n'
similar_html += '  </table>\n'
similar_html += '  <div class="sub" style="margin-top:8px;">根據當前觸發的因子組合，統計歷史上相同狀態後的大盤表現</div>\n'
similar_html += '</div>'

# ============================================================
# 2. 準備圖表資料 (JSON)
# ============================================================
# 加入真正的加權指數
if twii_real is not None:
    twii_display = twii_real.iloc[-1]
    # 用真正指數算回撤
    real_high = twii_real.expanding().max()
    real_dd = (twii_real / real_high - 1) * 100
    twii_dd_display = real_dd.iloc[-1]
else:
    twii_display = latest['twii']
    twii_dd_display = latest.get('drawdown', 0)

chart_data = {}
for col in ['abv_ma20', 'abv_ma60', 'rvol', 'pcr', 'margin_dd60', 'margin_bal_chg', 'drawdown']:
    s = df[col].dropna()
    chart_data[col] = {
        'dates': [d.strftime('%m/%d') for d in s.index],
        'values': [round(float(v), 2) for v in s.values],
    }

# 圖表上的指數用真正的加權指數
if twii_real is not None:
    real_recent = twii_real[twii_real.index >= start_date]
    chart_data['twii'] = {
        'dates': [d.strftime('%m/%d') for d in real_recent.index],
        'values': [round(float(v), 1) for v in real_recent.values],
    }
else:
    s = df['twii'].dropna()
    chart_data['twii'] = {
        'dates': [d.strftime('%m/%d') for d in s.index],
        'values': [round(float(v), 2) for v in s.values],
    }

# 台積 (稀疏, 用 ffill 後的)
tsmc_s = df['tsmc_chg'].dropna()
chart_data['tsmc_chg'] = {
    'dates': [d.strftime('%m/%d') for d in tsmc_s.index],
    'values': [round(float(v) * 100, 3) for v in tsmc_s.values],
}

# ============================================================
# 3. 生成 HTML
# ============================================================
print("Generating HTML...")

now_str = datetime.now().strftime('%Y-%m-%d %H:%M')

# 狀態顏色
def status_color(level):
    if level == 'danger': return '#ef4444'
    if level == 'warning': return '#f59e0b'
    if level == 'safe': return '#22c55e'
    return '#6b7280'

# 判斷整體狀態
if risk_v2:
    overall = ('danger', '撤退', '三重風險訊號觸發 (rvol+台積+PCR)')
elif risk_v1:
    overall = ('danger', '減碼', '台積PCR風險觸發')
elif a20 < 30:
    overall = ('warning', '偏空', f'寬度 {a20:.0f}%，市場弱勢')
elif a20 < 60:
    overall = ('warning', '留意', f'寬度 {a20:.0f}%，尚未轉強')
else:
    overall = ('safe', '正常', f'寬度 {a20:.0f}%，多頭格局')

# 抄底
any_bottom = any(bottom_signals.values())
if any_bottom:
    bottom_triggered = [k for k, v in bottom_signals.items() if v]
    bottom_status = ('safe', '抄底機會', ', '.join(bottom_triggered))
elif in_crash:
    bottom_status = ('warning', '大跌環境', f'回撤 {latest.get("drawdown", 0):.1f}%，等待訊號')
else:
    bottom_status = ('', '未在大跌', '')

html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>台股風險儀表板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 16px; }}
.header {{ text-align: center; margin-bottom: 20px; }}
.header h1 {{ font-size: 1.5em; color: #f8fafc; }}
.header .update {{ color: #94a3b8; font-size: 0.85em; margin-top: 4px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; margin-bottom: 16px; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; }}
.card h3 {{ color: #94a3b8; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
.card .value {{ font-size: 1.8em; font-weight: 700; }}
.card .sub {{ color: #94a3b8; font-size: 0.85em; margin-top: 4px; }}
.status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }}
.signal-card {{ border-width: 2px; }}
.chart-container {{ background: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; margin-bottom: 12px; }}
.chart-container h3 {{ color: #94a3b8; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }}
.chart-container canvas {{ max-height: 250px; }}
.row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }}
@media (max-width: 700px) {{ .row {{ grid-template-columns: 1fr; }} }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
th, td {{ padding: 6px 8px; text-align: right; border-bottom: 1px solid #334155; }}
th {{ color: #94a3b8; font-weight: 500; }}
td:first-child, th:first-child {{ text-align: left; }}
.dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
</style>
</head>
<body>

<div class="header">
  <h1>台股風險儀表板</h1>
  <div class="update">更新時間: {now_str}</div>
</div>

<!-- 主狀態 -->
<div class="grid">
  <div class="card signal-card" style="border-color: {status_color(overall[0])}">
    <h3>整體狀態</h3>
    <div class="value" style="color: {status_color(overall[0])}">{overall[1]}</div>
    <div class="sub">{overall[2]}</div>
  </div>
  <div class="card">
    <h3>加權指數</h3>
    <div class="value">{twii_display:,.0f}</div>
    <div class="sub">從高點回撤: {twii_dd_display:+.1f}% | 5日動能: {latest.get('mom5', 0):+.1f}%</div>
  </div>
  <div class="card">
    <h3>市場寬度</h3>
    <div class="value">{a20:.1f}%</div>
    <div class="sub">站上MA20比例 | MA60寬度: {a60:.1f}% | 差: {a20-a60:+.1f}%</div>
  </div>
</div>

<!-- 風險 & 抄底 -->
<div class="grid">
  <div class="card signal-card" style="border-color: {status_color('danger') if risk_v2 else (status_color('warning') if risk_v1 else status_color('safe'))}">
    <h3>風險迴避訊號</h3>
    <div>
      <span class="dot" style="background: {'#ef4444' if risk_v1 else '#22c55e'}"></span>
      v1 (台積+PCR): <strong>{'觸發!' if risk_v1 else '正常'}</strong>
    </div>
    <div style="margin-top: 6px;">
      <span class="dot" style="background: {'#ef4444' if risk_v2 else '#22c55e'}"></span>
      v2 (rvol+台積+PCR): <strong>{'觸發!' if risk_v2 else '正常'}</strong>
    </div>
  </div>
  <div class="card signal-card" style="border-color: {status_color(bottom_status[0]) if bottom_status[0] else '#334155'}">
    <h3>抄底訊號</h3>
    <div class="value" style="font-size: 1.2em; color: {status_color(bottom_status[0]) if bottom_status[0] else '#94a3b8'}">{bottom_status[1]}</div>
    <div class="sub">{bottom_status[2] if bottom_status[2] else '大盤回撤未達-10%門檻'}</div>
  </div>
</div>

<!-- 風險因子一覽表 -->
<div class="card" style="margin-bottom: 12px;">
  <h3>風險因子狀態</h3>
  <table>
    <tr><th style="text-align:left">因子</th><th>當前值</th><th>排名</th><th>門檻</th><th>狀態</th></tr>
    <tr>
      <td style="text-align:left">波動率</td>
      <td>{latest.get('rvol', 0):.1f}%</td>
      <td><strong>{int(latest.get('rvol_pct', 0)*100)}%</strong></td>
      <td>&ge;75%</td>
      <td style="color: {'#ef4444' if latest.get('rvol_pct', 0) >= 0.75 else '#22c55e'}">{'&#9745; 注意' if latest.get('rvol_pct', 0) >= 0.75 else '&#9744; 正常'}</td>
    </tr>
    <tr>
      <td style="text-align:left">台積大戶4w</td>
      <td>{latest.get('tsmc_chg', 0)*100:+.3f}%</td>
      <td><strong>{int(latest.get('tsmc_pct', 0.5)*100)}%</strong></td>
      <td>&le;25%</td>
      <td style="color: {'#ef4444' if latest.get('tsmc_pct', 0.5) <= 0.25 else '#22c55e'}">{'&#9745; 注意' if latest.get('tsmc_pct', 0.5) <= 0.25 else '&#9744; 正常'}</td>
    </tr>
    <tr>
      <td style="text-align:left">PCR未平倉</td>
      <td>{latest.get('pcr_raw', 0):.1f} <span style="color:#64748b; font-size:0.8em">(5日均:{latest.get('pcr', 0):.1f})</span></td>
      <td><strong>{int(latest.get('pcr_pct', 0)*100)}%</strong></td>
      <td>&le;30%</td>
      <td style="color: {'#ef4444' if latest.get('pcr_pct', 0) <= 0.30 else '#22c55e'}">{'&#9745; 注意' if latest.get('pcr_pct', 0) <= 0.30 else '&#9744; 正常'}</td>
    </tr>
  </table>
  <div class="sub" style="margin-top:10px; padding-top:8px; border-top:1px solid #334155; line-height:1.6;">
    三個同時超標 = 觸發風險迴避(v2)，歷史風險率 21%<br>
    波動率高 = 市場震盪 | 台積大戶撤 = 聰明錢在跑 | PCR低 = 散戶自滿沒買保險
  </div>
</div>

<!-- 歷史相似狀態統計 -->
{similar_html}

<!-- 融資 & 抄底 -->
<div class="card" style="margin-bottom: 12px;">
  <h3>融資 / 抄底指標</h3>
  <table>
    <tr><th style="text-align:left">指標</th><th>當前值</th><th>說明</th></tr>
    <tr>
      <td style="text-align:left">融資個股DD60</td>
      <td><strong>{latest.get('margin_dd60', 0):+.1f}%</strong></td>
      <td style="text-align:left; color: {'#ef4444' if latest.get('margin_dd60', 0) < -20 else '#f59e0b' if latest.get('margin_dd60', 0) < -15 else '#94a3b8'}">{'斷頭潮! 可抄底' if latest.get('margin_dd60', 0) < -20 else '嚴重套牢' if latest.get('margin_dd60', 0) < -15 else '一般'}</td>
    </tr>
    <tr>
      <td style="text-align:left">融資餘額20日變化</td>
      <td><strong>{latest.get('margin_bal_chg', 0):+.1f}%</strong></td>
      <td style="text-align:left">{'大量出逃' if latest.get('margin_bal_chg', 0) < -10 else '減少中' if latest.get('margin_bal_chg', 0) < -5 else '正常'}</td>
    </tr>
  </table>
</div>

<!-- 圖表 -->
<div class="row">
  <div class="chart-container">
    <h3>大盤指數 + 市場寬度</h3>
    <canvas id="chart1"></canvas>
  </div>
  <div class="chart-container">
    <h3>PCR + 台積電大戶4w變化</h3>
    <canvas id="chart2"></canvas>
  </div>
</div>
<div class="row">
  <div class="chart-container">
    <h3>融資個股DD60 + 大盤回撤</h3>
    <canvas id="chart3"></canvas>
  </div>
  <div class="chart-container">
    <h3>波動率 (已實現)</h3>
    <canvas id="chart4"></canvas>
  </div>
</div>

<script>
const D = {json.dumps(chart_data, ensure_ascii=False)};

const chartOpts = {{
  responsive: true,
  interaction: {{ mode: 'index', intersect: false }},
  plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#64748b', maxTicksLimit: 12, font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
    y: {{ ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ color: '#334155' }} }},
  }}
}};

function dualAxis(opts) {{
  return {{
    ...chartOpts,
    scales: {{
      ...chartOpts.scales,
      y: {{ position: 'left', ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ color: '#334155' }} }},
      y1: {{ position: 'right', ticks: {{ color: '#64748b', font: {{ size: 10 }} }}, grid: {{ drawOnChartArea: false }} }},
    }}
  }};
}}

// Chart 1: TWII + Breadth
new Chart(document.getElementById('chart1'), {{
  type: 'line',
  data: {{
    labels: D.twii.dates,
    datasets: [
      {{ label: '加權指數', data: D.twii.values, borderColor: '#3b82f6', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y' }},
      {{ label: 'AbvMA20%', data: D.abv_ma20.values, borderColor: '#22c55e', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y1' }},
      {{ label: 'AbvMA60%', data: D.abv_ma60.values, borderColor: '#6b7280', borderWidth: 1, pointRadius: 0, borderDash: [4,4], yAxisID: 'y1' }},
    ]
  }},
  options: dualAxis()
}});

// Chart 2: PCR + TSMC
new Chart(document.getElementById('chart2'), {{
  type: 'line',
  data: {{
    labels: D.pcr.dates,
    datasets: [
      {{ label: 'PCR未平倉', data: D.pcr.values, borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y' }},
      {{ label: '台積大戶4w(%)', data: D.tsmc_chg.values, borderColor: '#ef4444', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y1' }},
    ]
  }},
  options: dualAxis()
}});

// Chart 3: Margin DD + Drawdown
new Chart(document.getElementById('chart3'), {{
  type: 'line',
  data: {{
    labels: D.margin_dd60.dates,
    datasets: [
      {{ label: '融資DD60%', data: D.margin_dd60.values, borderColor: '#ef4444', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y' }},
      {{ label: '大盤回撤%', data: D.drawdown.values, borderColor: '#6366f1', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y1' }},
    ]
  }},
  options: dualAxis()
}});

// Chart 4: Volatility
new Chart(document.getElementById('chart4'), {{
  type: 'line',
  data: {{
    labels: D.rvol.dates,
    datasets: [
      {{ label: '已實現波動率%', data: D.rvol.values, borderColor: '#a855f7', borderWidth: 1.5, pointRadius: 0, fill: {{ target: 'origin', above: 'rgba(168,85,247,0.1)' }} }},
    ]
  }},
  options: chartOpts
}});
</script>

</body>
</html>"""

# 寫入
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"OK -> {output_path}")
print(f"Date: {latest_date.strftime('%Y-%m-%d')}")
print(f"Status: {overall[1]} | Risk v2={'ON' if risk_v2 else 'off'}")
