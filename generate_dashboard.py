"""
市場風險儀表板 — 生成靜態 HTML
每天跑一次，生成 index.html 供 GitHub Pages 部署
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd, numpy as np, json, warnings, gc
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
    calc_breadth, _expanding_pct, _load_rvol,
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
tsmc_chg, tsmc_pct = _load_tsmc_holder()
margin_dd60 = _load_margin_dd60()
margin_bal_chg = _load_margin_bal_chg()

# === 2026-08 驗證新增 ===
# HL20 組合減碼訊號: 全市場 NH20/(NH20+NL20) 10日均 <50% -> 半倉 (2012-18 + 2019-26 雙窗口驗證)
_ca = data.get('etl:adj_close')
_cols4 = [x for x in _ca.columns if len(x) == 4 and x.isdigit() and not x.startswith('00')]
_ca = _ca[_cols4]
_nh20 = (_ca >= _ca.rolling(20).max()).sum(axis=1)
_nl20 = (_ca <= _ca.rolling(20).min()).sum(axis=1)
hl20 = (_nh20 / (_nh20 + _nl20).replace(0, np.nan)).rolling(10).mean() * 100

# 融資背離占比: 融資20日增>5% 且 股價20日跌>5% 的個股占比 (>10% = 斷頭潮預備警報)
_mbal = data.get('margin_transactions:融資今日餘額')
_craw = data.get('price:收盤價')
_mcols = [x for x in _cols4 if x in _mbal.columns and x in _craw.columns]
_mb = _mbal[_mcols]
_cr = _craw[_mcols]
_div = (_mb.pct_change(20) > 0.05) & (_cr.pct_change(20) < -0.05) & (_mb > 0)
margin_div = _div.sum(axis=1) / (_mb > 0).sum(axis=1) * 100

# 融資維持率 (2026-08 新增, XQ官方算法, 驗證見 bottom_margin_maint.py):
# 個股融資成本_t = (昨日成本x(餘額-買進) + 收盤x買進) / 餘額; 維持率 = 收盤/(成本x0.6)x100
# pct130 = 維持率<130%家數占比; >25% = 斷頭潮抄底訊號 (2020起6事件100%勝率)
_mbuy = data.get('margin_transactions:融資買進')
_mm_cols = [x for x in _mcols if x in _mbuy.columns]
_mm_idx = _mbal[_mm_cols].loc['2015-01-01':].index  # 成本線自2015暖機
_B = _mbuy[_mm_cols].reindex(_mm_idx).fillna(0).values
_L = _mbal[_mm_cols].reindex(_mm_idx).fillna(0).values
_P = _craw[_mm_cols].reindex(_mm_idx).ffill().values
_cost = np.full(len(_mm_cols), np.nan)
_p130 = np.full(len(_mm_idx), np.nan)
_mavg = np.full(len(_mm_idx), np.nan)
for _t in range(len(_mm_idx)):
    _b, _l, _p = _B[_t], _L[_t], _P[_t]
    _carried = np.maximum(_l - _b, 0)
    _new = np.where(_l > 0, (np.nan_to_num(_cost) * _carried + _p * _b) / np.where(_l > 0, _l, 1), np.nan)
    _new = np.where(np.isnan(_cost) & (_l > 0), _p, _new)  # 餘額歸零後重啟: 成本=當日價
    _cost = np.where(_l > 0, _new, np.nan)
    _v = (_l > 0) & ~np.isnan(_cost) & (_cost > 0) & ~np.isnan(_p)
    if _v.sum() > 0:
        _m = _p[_v] / (_cost[_v] * 0.6) * 100
        _p130[_t] = (_m < 130).mean() * 100
        _mavg[_t] = (_p[_v] * _l[_v]).sum() / (_cost[_v] * 0.6 * _l[_v]).sum() * 100
pct130 = pd.Series(_p130, index=_mm_idx)
maint_avg = pd.Series(_mavg, index=_mm_idx)

_p130_valid = pct130.dropna()
pct130_now = _p130_valid.iloc[-1]
maint_now = maint_avg.dropna().iloc[-1]
_p130_hist5y = _p130_valid[_p130_valid.index >= _p130_valid.index[-1] - pd.Timedelta(days=365 * 5)]
pct130_rank_now = (_p130_hist5y <= pct130_now).mean() * 100
pct130_peak20 = _p130_valid.rolling(20).max().iloc[-1]
pct130_ebb = (pct130_peak20 > 30) and (pct130_now < pct130_peak20 * 2 / 3)  # 退潮=斷頭出盡確認

# === S1/S2 領先風險訊號 (2026-08 新增) ===
# S1 CH_DEF_RS 防禦輪動: 高檔區 in-sample 6.12x + OOS 2014-19 8.29x 雙窗口通過 (rotation_lead.py / oos_rotation.py)
# S2 熱冷輪動危險型: 2020後 regime 訊號 (前半14.3x/後半2.3x, OOS前失效), 輔訊號, 連3次假警報除役
import ast as _ast
_amtall = data.get('price:成交金額')
_scols = [c for c in _cols4 if c in _amtall.columns]
_amtS = _amtall[_scols]
_adjS = _ca[_scols]
_liqm = _amtS.rolling(20).mean() > 3e7
_th2 = data.get('security_industry_themes')
_members = {}
for _, _r in _th2.iterrows():
    _s = str(_r['stock_id'])
    if _s not in _scols: continue
    try: _cats = _ast.literal_eval(_r['category'])
    except Exception: continue
    for _c in _cats: _members.setdefault(_c.split(':')[0].strip(), set()).add(_s)
_CH = {k: sorted(v) for k, v in _members.items() if len(v) >= 8}
_BD = ['食品', '食品生技', '金融', '金融科技', '水泥', '油電燃氣', '造紙', '貿易百貨']
_BT = ['半導體', '電腦及週邊設備', '通信網路', '被動元件', '印刷電路板', '平面顯示器', '觸控面板',
       '連接器', 'LED照明產業', '人工智慧', '雲端運算', '軟體服務', '資通訊安全', '大數據', '能源元件']
_BD = [c for c in _BD if c in _CH]; _BT = [c for c in _BT if c in _CH]
_dset = set().union(*[set(_CH[c]) for c in _BD]); _tset = set().union(*[set(_CH[c]) for c in _BT])
_ovl = _dset & _tset
_dgrp = {c: [x for x in _CH[c] if x not in _ovl] for c in _BD}
_tgrp = {c: [x for x in _CH[c] if x not in _ovl] for c in _BT}
_dgrp = {k: v for k, v in _dgrp.items() if len(v) >= 5}; _tgrp = {k: v for k, v in _tgrp.items() if len(v) >= 5}
def _chret(dic):
    return pd.DataFrame({k: (_adjS[v] / _adjS[v].shift(20) - 1)[_liqm[v]].median(axis=1) * 100 for k, v in dic.items()})
ch_def_rs = _chret(_dgrp).median(axis=1) - _chret(_tgrp).median(axis=1)

_boom = (_adjS.shift(20) / _adjS.shift(120) - 1).where(_liqm)
_recent = _adjS / _adjS.shift(20) - 1
_amt5S = _amtS.rolling(5).mean()
_shr = _amt5S.div(_amt5S.sum(axis=1), axis=0) * 100
_dshr = _shr - _shr.shift(20)
_bv = _boom.values; _rv = _recent.values; _dvv = _dshr.values
_retA = np.full(len(_adjS), np.nan); _flwA = np.full(len(_adjS), np.nan)
for _i in range(140, len(_adjS)):
    _b = _bv[_i]; _ok = ~np.isnan(_b)
    if _ok.sum() < 150: continue
    _q = np.nanquantile(_b[_ok], [0.2, 0.8])
    _lag = _ok & (_b <= _q[0]); _hot = _ok & (_b >= _q[1])
    _retA[_i] = (np.nanmedian(_rv[_i][_lag]) - np.nanmedian(_rv[_i][_hot])) * 100
    _flwA[_i] = np.nansum(_dvv[_i][_lag]) - np.nansum(_dvv[_i][_hot])
ret_l2h = pd.Series(_retA, index=_adjS.index); flw_l2h = pd.Series(_flwA, index=_adjS.index)

def _exp_rank(s):
    v = s.values; out = np.full(len(v), np.nan)
    for i in range(250, len(v)):
        p = v[:i]; p = p[~np.isnan(p)]
        if len(p) > 100 and not np.isnan(v[i]): out[i] = (p <= v[i]).mean() * 100
    return pd.Series(out, index=s.index)
ch_def_rank = _exp_rank(ch_def_rs)
ret_rank = _exp_rank(ret_l2h)
flw_rank = _exp_rank(flw_l2h)

# S3 = 熱門群大戶撤出 (週頻集保, 2026-08 新增): 唯一通過雙窗口的籌碼群組因子
# IS 1.81x / OOS 3.82x, 延遲3日仍活但半衰期短; 與台積大戶相關僅0.07 (a3_robust.py)
s3_ok = False
a3_val_now = np.nan; a3_rank_now = np.nan; a3_week_date = None
a3_rank_series = pd.Series(dtype=float)
try:
    _liq2 = _amtS.rolling(20).mean() > 2e7
    _brank2 = (_adjS.shift(20) / _adjS.shift(120) - 1).where(_liq2).rank(axis=1, pct=True)
    _hot2 = _brank2 >= 0.8
    _inv = data.get('inventory').reset_index()
    _inv['lv'] = _inv['持股分級'].astype(int)
    _big = _inv[_inv.lv.between(12, 15)].groupby(['date', 'stock_id'])['持有股數'].sum().unstack()
    _tot = _inv.groupby(['date', 'stock_id'])['持有股數'].sum().unstack()
    del _inv; gc.collect()
    _ratioA = (_big / _tot * 100)
    _ratioA.index = pd.to_datetime(_ratioA.index)
    _ratioA = _ratioA[[c for c in _ratioA.columns if c in _scols]]
    _chg4wA = _ratioA - _ratioA.shift(4)
    _hotwA = _hot2[_ratioA.columns].reindex(_chg4wA.index, method='ffill')
    a3_week = _chg4wA.where(_hotwA).median(axis=1).dropna()
    a3_daily = a3_week.reindex(_adjS.index).ffill()
    a3_rank_series = _exp_rank(a3_daily)
    a3_val_now = float(a3_week.iloc[-1])
    a3_rank_now = float(a3_rank_series.dropna().iloc[-1])
    a3_week_date = a3_week.index[-1]
    del _big, _tot, _ratioA, _chg4wA; gc.collect()
    s3_ok = True
except Exception as _e:
    print(f'  S3 (A3) failed: {type(_e).__name__}: {_e}')

_twdd_now = (twii / twii.cummax() - 1).iloc[-1] * 100
lead_armed = _twdd_now > -5  # 訊號只在高檔區(回撤<5%)驗證有效

# === 紅燈閂鎖 (2026-08-06): 紅燈觸發後不因跌破-5%解除 ===
# 釋放: ①S1<P50 (假警報解除) 或 ②回撤<=-10%且pct130>25 (移交抄底系統)
# 驗證: latch_test.py — combo4 Calmar 8.36→11.2, MDD -18.1→-13.3; P70釋放線會在2026-07-13險些提前解鎖, 故用P50
_twdd_full = (twii / twii.cummax() - 1) * 100
_s1_series = (ch_def_rank >= 90) & (_twdd_full.reindex(ch_def_rank.index) > -5)
_s2_series = (ret_rank >= 80) & (flw_rank <= 50) & (_twdd_full.reindex(ret_rank.index) > -5)
_s3_series = (a3_rank_series <= 20) & (_twdd_full.reindex(a3_rank_series.index) > -5) if not a3_rank_series.dropna().empty else pd.Series(False, index=ch_def_rank.index)
_p130_daily = pct130.reindex(ch_def_rank.index).ffill()
latch_state = None  # None / 0.3 / 0.5
latch_since = None
latch_handoff = False
for _d in ch_def_rank.index[ch_def_rank.index >= '2020-01-01']:
    _s1 = bool(_s1_series.get(_d, False)); _s2 = bool(_s2_series.get(_d, False)); _s3 = bool(_s3_series.get(_d, False))
    if latch_state is None:
        if _s1 and _s2:
            latch_state = 0.3; latch_since = _d; latch_handoff = False
        elif _s1 and _s3:
            latch_state = 0.5; latch_since = _d; latch_handoff = False
    else:
        if _s1 and _s2:
            latch_state = 0.3
        _r = ch_def_rank.get(_d, np.nan)
        _ddv = _twdd_full.reindex([_d]).iloc[0] if _d in _twdd_full.index else np.nan
        _pv = _p130_daily.get(_d, np.nan)
        if not np.isnan(_r) and _r < 50:
            latch_state = None; latch_handoff = False
        elif not np.isnan(_ddv) and _ddv <= -10 and not np.isnan(_pv) and _pv > 25:
            latch_state = None; latch_handoff = True  # 移交抄底系統
s1_val_now = float(ch_def_rs.dropna().iloc[-1]); s1_rank_now = float(ch_def_rank.dropna().iloc[-1])
ret_rank_now = float(ret_rank.dropna().iloc[-1]); flw_rank_now = float(flw_rank.dropna().iloc[-1])
s1_on = lead_armed and s1_rank_now >= 90
s2_on = lead_armed and ret_rank_now >= 80 and flw_rank_now <= 50
s3_on = lead_armed and s3_ok and a3_rank_now <= 20
double_low = lead_armed and s1_rank_now < 30 and ret_rank_now < 30  # 雙低 = 壓重倉環境 (+5.3%/20日, 83%勝率)
n_lit = int(s1_on) + int(s2_on) + int(s3_on)
_lit_names = " ".join(n for n, on in [('S1', s1_on), ('S2', s2_on), ('S3', s3_on)] if on)
if latch_state is not None:
    lead_action = ('danger', f'紅燈閂鎖 {latch_state}x', f'{latch_since.date()} 觸發後鎖定（不因跌破-5%解除）— 釋放: S1<P50 或 回撤≤-10%且pct130>25移交抄底')
elif not lead_armed and latch_handoff:
    lead_action = ('warning', '已移交抄底系統', f'紅燈閂鎖已在崩跌區釋放（pct130>25 斷頭潮確認）— 依三段建倉 playbook 行動 | 回撤 {_twdd_now:.1f}%')
elif not lead_armed:
    lead_action = ('', '未武裝', f'大盤回撤 {_twdd_now:.1f}% 已超過-5%，領先訊號不適用，由抄底系統接管')
elif s1_on and s2_on:
    lead_action = ('danger', '減碼至 0.3x', f'{_lit_names} 同亮（S1+S2 強紅燈: 精確率100%, 2021-05/2026-06 兩役皆中）')
elif s1_on and s3_on:
    lead_action = ('danger', '減碼至 0.5x', 'S1+S3 中信心紅燈（OOS 23x 但近年薄弱、2024-05曾誤報 → 半減, S2加入才0.3x）')
elif n_lit >= 2:
    lead_action = ('warning', '警戒（不動作）', f'{_lit_names} 同亮但無S1 — 籌碼互相壯膽=假合唱（歷史10事件0命中），等價格面確認')
elif n_lit == 1 and s3_on and double_low:
    lead_action = ('safe', '滿倉 1.0x（不加壓）', '雙低但S3亮 — 仍是順風（+3.5%/20日, 80%勝, 0崩盤）, 只是不升1.5x')
elif n_lit == 1:
    lead_action = ('warning', '警戒（不動作）', f'{_lit_names} 單獨亮 — 單訊號=噪音（OOS 0.00x），等確認')
elif double_low:
    lead_action = ('safe', '重壓檔 1.5x', '雙低：S1<P30 且 RET<P30 — 主流領漲、防禦死寂（歷史+5.3%/20日, 83%勝率）')
else:
    lead_action = ('safe', '滿倉 1.0x', '領先訊號正常')
s3_val_txt = f"{a3_val_now:+.2f}" if s3_ok else "n/a"
s3_rank_txt = f"P{a3_rank_now:.0f}" if s3_ok else "-"
s3_date_txt = str(a3_week_date.date()) if s3_ok and a3_week_date is not None else "資料失敗"

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
    'mom5': mom5,
    'margin_dd60': margin_dd60,
    'margin_bal_chg': margin_bal_chg,
    'hl20': hl20,
    'margin_div': margin_div,
    'pct130': pct130,
    'maint_avg': maint_avg,
    'ch_def_rank': ch_def_rank,
    'ret_rank': ret_rank,
    'flw_rank': flw_rank,
    'a3_rank': a3_rank_series,
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

# 風險訊號: rvol+台積 二取二 (2026-08 改版, 移除PCR — 避險型崩盤盲區)
risk_rt2 = (latest.get('rvol_pct', 0.5) >= 0.75) and (latest.get('tsmc_pct', 0.5) <= 0.25)

# 抄底
in_crash = latest.get('drawdown', 0) <= -10
# 融資類資料晚間才發布, 當日可能 NaN → 取最後有效值 (否則訊號會靜默失效)
def _lastvalid(col, default=0):
    s = df[col].dropna()
    return float(s.iloc[-1]) if len(s) else default
mdd60 = _lastvalid('margin_dd60')
mbal20 = _lastvalid('margin_bal_chg')
bottom_signals = {
    '融資DD<-20%(斷頭潮)': in_crash and mdd60 < -20,
    '融資DD<-15%': in_crash and mdd60 < -15,
    '融資DD<-15%+動能>0': in_crash and mdd60 < -15 and latest.get('mom5', 0) > 0,
    'rvol>P90': in_crash and latest.get('rvol_pct', 0) >= 0.90,
    '維持率<130%家數>25%': in_crash and pct130_now > 25,
    '融資餘額20d減>10%': in_crash and mbal20 < -10,
}

# HL20 減碼訊號 & 融資背離
hl20_val = latest.get('hl20', 50)
hl20_half = hl20_val < 50
mdiv_val = _lastvalid('margin_div')

# ============================================================
# 1b. 歷史相似狀態統計 (2020起)
# ============================================================
print("  calculating similar historical states...")

# 用全期間的資料
df_full = pd.DataFrame({
    'twii': twii, 'rvol_pct': rvol_pct,
}).dropna()
df_full['tsmc_pct'] = tsmc_pct.reindex(df_full.index, method='ffill')
df_full = df_full[df_full.index >= '2020-01-01'].dropna()

for fwd in [5, 10, 20]:
    df_full[f'fwd_{fwd}d'] = twii.pct_change(fwd).shift(-fwd).reindex(df_full.index) * 100

# 統計各組合
similar_stats = []

# 目前狀態: 用 ±10% 範圍匹配「跟現在類似的歷史日」
rv = latest.get('rvol_pct', 0.5)
tv = latest.get('tsmc_pct', 0.5)

combos = [
    (f'目前狀態 (波動P{int(rv*100)} 台積P{int(tv*100)})', [
        ('rvol_pct', '>=', max(0, rv - 0.10)),
        ('rvol_pct', '<=', min(1, rv + 0.10)),
        ('tsmc_pct', '>=', max(0, tv - 0.10)),
        ('tsmc_pct', '<=', min(1, tv + 0.10)),
    ]),
]

# 固定的危險因子 (不管當前有沒有觸發都列)
danger_factors = [
    ('波動率高(>=P75)', 'rvol_pct', '>=', 0.75),
    ('台積大戶撤(<=P25)', 'tsmc_pct', '<=', 0.25),
]

def is_triggered(col, op, th):
    v = latest.get(col, 0.5)
    return (op == '>=' and v >= th) or (op == '<=' and v <= th)

# 各單因子
for name, col, op, th in danger_factors:
    tag = ' ★當前符合' if is_triggered(col, op, th) else ''
    combos.append((f'{name}{tag}', [(col, op, th)]))

# 二取二 = 風險迴避訊號
all2 = all(is_triggered(c, o, t) for _, c, o, t in danger_factors)
combos.append((f'波動高+台積撤=風險迴避{" ★當前符合" if all2 else ""}', [
    ('rvol_pct', '>=', 0.75), ('tsmc_pct', '<=', 0.25)
]))

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
similar_html += '  <div class="sub" style="margin-top:8px; line-height:1.6;">'
similar_html += '「目前狀態」= 兩個因子都在當前值 &plusmn;10% 範圍內的歷史日<br>'
similar_html += '其餘為固定危險門檻的回測 | ★ = 當前符合該條件'
similar_html += '</div>\n'
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
for col in ['abv_ma20', 'abv_ma60', 'rvol', 'margin_dd60', 'margin_bal_chg', 'drawdown', 'hl20', 'margin_div', 'pct130', 'maint_avg', 'ch_def_rank', 'ret_rank', 'flw_rank', 'a3_rank']:
    if col == 'a3_rank' and df[col].dropna().empty:
        continue
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
if risk_rt2:
    overall = ('danger', '撤退', '風險訊號觸發 (波動高+台積大戶撤)')
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

<!-- ★ 領先風險訊號 (2026-08 最新驗證, 高檔區限定) -->
<div class="grid">
  <div class="card signal-card" style="border-color: {status_color(lead_action[0]) if lead_action[0] else '#334155'}">
    <h3>★ 領先訊號行動 (combo4 曝險)</h3>
    <div class="value" style="font-size: 1.4em; color: {status_color(lead_action[0]) if lead_action[0] else '#94a3b8'}">{lead_action[1]}</div>
    <div class="sub">{lead_action[2]}<br>檔位: 雙低→1.5x | 正常→1.0x | <strong>S1+S2→0.3x鎖 | S1+S3→0.5x鎖</strong> | 其餘亮法→僅警戒<br><span style="color:#64748b">紅燈=閂鎖制: 觸發後不因跌破-5%解除, 直到S1&lt;P50(假警報)或崩跌區pct130&gt;25(移交抄底)。單訊號=噪音; S2+S3無S1=假合唱; 事件型崩跌由HL20+策略濾網兜底</span></div>
  </div>
  <div class="card signal-card" style="border-color: {'#ef4444' if s1_on else '#22c55e' if lead_armed else '#334155'}">
    <h3>S1 防禦輪動 CH_DEF_RS</h3>
    <div class="value" style="font-size: 1.4em">{s1_val_now:+.1f} <span style="font-size:0.7em; color:{'#ef4444' if s1_rank_now>=90 else '#94a3b8'}">P{s1_rank_now:.0f}</span></div>
    <div class="sub">防禦8鏈 - 科技15鏈 20日報酬差 | 減碼線 P90 / 重壓線 P30{' | ⚠未武裝' if not lead_armed else ''}<br>驗證: 高檔區 6.1x + OOS 2014-19 8.3x (雙窗口)</div>
  </div>
  <div class="card signal-card" style="border-color: {'#ef4444' if s2_on else '#22c55e' if lead_armed else '#334155'}">
    <h3>S2 熱冷輪動 (輔訊號)</h3>
    <div class="value" style="font-size: 1.3em">RET P{ret_rank_now:.0f} <span style="color:{'#ef4444' if ret_rank_now >= 80 else '#22c55e'}">{'✓' if ret_rank_now >= 80 else '✗'}</span> ｜ FLW P{flw_rank_now:.0f} <span style="color:{'#ef4444' if flw_rank_now <= 50 else '#22c55e'}">{'✓' if flw_rank_now <= 50 else '✗'}</span></div>
    <div class="sub"><strong>亮 = RET&ge;P80 且 FLW&le;P50 兩腳同時成立</strong>（缺一不可）{' | ⚠未武裝' if not lead_armed else ''}<br>RET高=冷門跑贏(主流熄火) | FLW高=資金真搬家(健康換手✗) / FLW低=熱門跌但資金不撤(出貨✓)<br>2020後 regime 訊號 (14.3x/2.3x) | 連3次假警報自動除役</div>
  </div>
  <div class="card signal-card" style="border-color: {'#ef4444' if s3_on else '#22c55e' if lead_armed and s3_ok else '#334155'}">
    <h3>S3 大戶撤出熱門群 (週頻)</h3>
    <div class="value" style="font-size: 1.4em">{s3_val_txt} <span style="font-size:0.7em; color:{'#ef4444' if s3_ok and a3_rank_now<=20 else '#94a3b8'}">{s3_rank_txt}</span></div>
    <div class="sub">熱門群(120日漲幅前20%)大戶&ge;400張持股比 4週變化中位數 | 危險 &le;P20{' | ⚠未武裝' if not lead_armed else ''}<br>驗證: IS 1.8x + OOS 3.8x (雙窗口) | 資料週 {s3_date_txt} | 半衰期短, 週更即用</div>
  </div>
  <div class="card signal-card" style="border-color: {'#22c55e' if pct130_ebb else '#ef4444' if pct130_now > 25 else '#334155'}">
    <h3>斷頭潮 pct130 (抄底系統)</h3>
    <div class="value" style="font-size: 1.4em">{pct130_now:.1f}% <span style="font-size:0.7em; color:#94a3b8">5年P{pct130_rank_now:.0f}</span></div>
    <div class="sub">{'退潮確認! 斷頭出盡' if pct130_ebb else '斷頭潮! 抄底訊號 (>25%)' if pct130_now > 25 else '維持率<130%家數占比 | 抄底門檻 25%'}<br>驗證: 回撤&ge;10%環境 6事件 100%勝</div>
  </div>
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
  <div class="card signal-card" style="border-color: {status_color('danger') if risk_rt2 else status_color('safe')}">
    <h3>風險迴避訊號</h3>
    <div>
      <span class="dot" style="background: {'#ef4444' if risk_rt2 else '#22c55e'}"></span>
      波動高 + 台積大戶撤 (二取二): <strong>{'觸發!' if risk_rt2 else '正常'}</strong>
    </div>
    <div class="sub" style="margin-top: 6px;">rvol&ge;P75 且 台積大戶4w&le;P25</div>
  </div>
  <div class="card signal-card" style="border-color: {status_color(bottom_status[0]) if bottom_status[0] else '#334155'}">
    <h3>抄底訊號</h3>
    <div class="value" style="font-size: 1.2em; color: {status_color(bottom_status[0]) if bottom_status[0] else '#94a3b8'}">{bottom_status[1]}</div>
    <div class="sub">{bottom_status[2] if bottom_status[2] else '大盤回撤未達-10%門檻'}</div>
  </div>
  <div class="card signal-card" style="border-color: {'#f59e0b' if hl20_half else '#22c55e'}">
    <h3>組合減碼訊號 HL20</h3>
    <div class="value" style="font-size: 1.2em; color: {'#f59e0b' if hl20_half else '#22c55e'}">{'半倉' if hl20_half else '滿倉'}</div>
    <div class="sub">新高/(新高+新低) 10日均 = {hl20_val:.1f}%（&lt;50% 減碼一半）<br>2012-18 + 2019-26 雙窗口驗證</div>
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
  </table>
  <div class="sub" style="margin-top:10px; padding-top:8px; border-top:1px solid #334155; line-height:1.6;">
    兩個同時超標 = 觸發風險迴避（二取二）<br>
    波動率高 = 市場震盪 | 台積大戶撤 = 聰明錢在跑<br>
    <span style="color:#64748b">註: PCR 已於 2026-08 移除 — 避險型崩盤(如2026/07)全程不反應, 結構性盲區</span>
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
      <td><strong>{mdd60:+.1f}%</strong></td>
      <td style="text-align:left; color: {'#ef4444' if mdd60 < -20 else '#f59e0b' if mdd60 < -15 else '#94a3b8'}">{'斷頭潮! 可抄底' if mdd60 < -20 else '嚴重套牢' if mdd60 < -15 else '一般'}</td>
    </tr>
    <tr>
      <td style="text-align:left">融資餘額20日變化</td>
      <td><strong>{mbal20:+.1f}%</strong></td>
      <td style="text-align:left">{'大量出逃' if mbal20 < -10 else '減少中' if mbal20 < -5 else '正常'}</td>
    </tr>
    <tr>
      <td style="text-align:left">融資戶平均維持率</td>
      <td><strong>{maint_now:.1f}%</strong></td>
      <td style="text-align:left; color: {'#ef4444' if maint_now < 140 else '#f59e0b' if maint_now < 150 else '#94a3b8'}">{'逼近130追繳線!' if maint_now < 140 else '低於150警戒線' if maint_now < 150 else '正常 (滿手成本約166%)'}</td>
    </tr>
    <tr>
      <td style="text-align:left">維持率&lt;130%家數占比</td>
      <td><strong>{pct130_now:.1f}%</strong> (5年P{pct130_rank_now:.0f})</td>
      <td style="text-align:left; color: {'#22c55e' if pct130_ebb else '#ef4444' if pct130_now > 25 else '#f59e0b' if pct130_now > 15 else '#94a3b8'}">{'退潮確認! 斷頭出盡 (峰值' + f'{pct130_peak20:.0f}' + '%回落1/3)' if pct130_ebb else '斷頭潮! 抄底訊號 (>25%, 6/6勝)' if pct130_now > 25 else '追繳壓力升高' if pct130_now > 15 else '正常'}</td>
    </tr>
    <tr>
      <td style="text-align:left">融資背離占比</td>
      <td><strong>{mdiv_val:.1f}%</strong></td>
      <td style="text-align:left; color: {'#ef4444' if mdiv_val > 10 else '#f59e0b' if mdiv_val > 7 else '#94a3b8'}">{'火藥庫滿載! 斷頭潮預備(短線防最後一殺, 20日後多反彈)' if mdiv_val > 10 else '套牢盤堆積中' if mdiv_val > 7 else '正常'}</td>
    </tr>
  </table>
  <div class="sub" style="margin-top:8px;">背離占比 = 融資20日增&gt;5% 且 股價20日跌&gt;5% 的個股占比；&gt;10% 史上僅2%的日子，通常是斷頭潮前奏<br>
  維持率用 XQ 官方算法推算 (成本遞迴自2015暖機, 上市櫃4碼普通股)；&lt;130%家數占比&gt;25% = 斷頭潮抄底 (2020起6事件後20日全勝)，峰值&gt;30%後回落1/3 = 賣壓出盡確認</div>
</div>

<!-- 圖表 -->
<div class="row">
  <div class="chart-container">
    <h3>大盤指數 + 市場寬度</h3>
    <canvas id="chart1"></canvas>
  </div>
  <div class="chart-container">
    <h3>台積電大戶4w變化 + 融資背離占比</h3>
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
<div class="row" style="grid-template-columns: 1fr;">
  <div class="chart-container">
    <h3>融資維持率壓力 — &lt;130%家數占比 (&gt;25%=斷頭潮抄底) + 平均維持率</h3>
    <canvas id="chart5"></canvas>
  </div>
</div>
<div class="row" style="grid-template-columns: 1fr;">
  <div class="chart-container">
    <h3>領先訊號歷史 (expanding 百分位) — 紅虛線P90=減碼 / 綠虛線P30=S1與RET雙低可重壓 / S3(紫,週頻)跌破P20=大戶撤出</h3>
    <canvas id="chart6"></canvas>
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
      {{ label: 'HL20減碼線%', data: D.hl20.values, borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y1' }},
    ]
  }},
  options: dualAxis()
}});

// Chart 2: TSMC + 融資背離占比
new Chart(document.getElementById('chart2'), {{
  type: 'line',
  data: {{
    labels: D.margin_div.dates,
    datasets: [
      {{ label: '融資背離占比%', data: D.margin_div.values, borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y' }},
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

// Chart 6: 領先訊號百分位
new Chart(document.getElementById('chart6'), {{
  type: 'line',
  data: {{
    labels: D.ch_def_rank.dates,
    datasets: [
      {{ label: 'S1 CH_DEF_RS百分位', data: D.ch_def_rank.values, borderColor: '#ef4444', borderWidth: 1.8, pointRadius: 0 }},
      {{ label: 'S2 RET百分位', data: D.ret_rank.values, borderColor: '#f59e0b', borderWidth: 1.3, pointRadius: 0 }},
      {{ label: 'S2 FLW百分位', data: D.flw_rank.values, borderColor: '#3b82f6', borderWidth: 1.3, pointRadius: 0 }},
      ...(D.a3_rank ? [{{ label: 'S3 大戶撤出百分位(週)', data: D.a3_rank.values, borderColor: '#a855f7', borderWidth: 1.5, pointRadius: 0 }}] : []),
      {{ label: 'P90減碼線', data: D.ch_def_rank.values.map(() => 90), borderColor: '#ef4444', borderWidth: 1, pointRadius: 0, borderDash: [4,4] }},
      {{ label: 'P30重壓線 (S1+RET雙低→1.5x)', data: D.ch_def_rank.values.map(() => 30), borderColor: '#22c55e', borderWidth: 1.2, pointRadius: 0, borderDash: [6,4] }},
    ]
  }},
  options: chartOpts
}});

// Chart 5: 融資維持率壓力
new Chart(document.getElementById('chart5'), {{
  type: 'line',
  data: {{
    labels: D.pct130.dates,
    datasets: [
      {{ label: '<130%家數占比%', data: D.pct130.values, borderColor: '#ef4444', borderWidth: 1.5, pointRadius: 0, fill: {{ target: 'origin', above: 'rgba(239,68,68,0.12)' }}, yAxisID: 'y' }},
      {{ label: '抄底門檻25%', data: D.pct130.values.map(() => 25), borderColor: '#ef4444', borderWidth: 1, pointRadius: 0, borderDash: [4,4], yAxisID: 'y' }},
      {{ label: '平均維持率%', data: D.maint_avg.values, borderColor: '#22d3ee', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y1' }},
      {{ label: '追繳線130%', data: D.maint_avg.values.map(() => 130), borderColor: '#22d3ee', borderWidth: 1, pointRadius: 0, borderDash: [4,4], yAxisID: 'y1' }},
    ]
  }},
  options: dualAxis()
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
print(f"Status: {overall[1]} | Risk RT2={'ON' if risk_rt2 else 'off'}")
