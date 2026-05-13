#!/usr/bin/env python3
import re, json, requests, sys
from datetime import datetime, timezone, timedelta
from scipy.stats import norm
import numpy as np

KT_TO_MPH = 1.15078
STATION    = "KRNO"
NOMADS_NBM = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"

_MATRIX = {
    4: [1, 2, 3, 4, 5],   # >90%
    3: [1, 2, 3, 4, 4],   # >66%
    2: [1, 2, 2, 3, 4],   # >33%
    1: [1, 1, 2, 2, 3],   # >10%
    0: [1, 1, 1, 2, 2],   # <10%
}

def risk_matrix(prob, level):
    if level == 0 or prob == 0: return 0
    pr = 4 if prob>=90 else 3 if prob>=66 else 2 if prob>=33 else 1 if prob>=10 else 0
    return _MATRIX[pr][level - 1]

RISK_C = {0:"#3f3f46",1:"#e2f0cb",2:"#ffeb3b",3:"#ff9800",4:"#f44336",5:"#9c27b0"}
RISK_L = {0:"NONE",1:"LITTLE TO NONE",2:"MINOR",3:"MODERATE",4:"MAJOR",5:"EXTREME"}
HAZARDS = ["WIND","LIGHTNING","SNOW","VISIBILITY","FZRA","FLASH_FREEZE","RAIN","TEMPERATURE"]

METRICS = {
    "WIND":        {2:"30-45 mph",    3:"45-58 mph",    4:"58-65 mph",   5:">65 mph"},
    "SNOW":        {2:"T-0.5 in/hr",  3:"0.5-1 in/hr",  4:"1-2 in/hr",   5:">2 in/hr"},
    "LIGHTNING":   {1:"<5%",  2:"5-25%",        3:"25-50%",       4:"50-75%",      5:">75%"},
    "VISIBILITY":  {2:"3-5 SM",       3:"1-3 SM",       4:"0.5-1 SM",    5:"<0.5 SM"},
    "RAIN":        {2:">0.10 in/hr",  3:">0.25 in/hr",  4:">0.50 in/hr", 5:">1.0 in/hr"},
    "FZRA":        {2:"Trace",        3:"Trace-0.01in", 4:"0.01-0.10in", 5:">0.10in"},
    "FLASH_FREEZE":{2:"Wet+Tw<36F",   3:"Wet+Tw<32F",   4:"Wet+Tw<28F",  5:"Wet+Tw<25F"},
    "HEAT":        {1:"85-90F",  2:"90-95F",       3:"95-100F",      4:"100-105F",    5:">105F"},
    "COLD":        {2:"32-40F",       3:"20-32F",       4:"10-20F",      5:"<10F"},
}

def gauss_above(mean, std, thr):
    if mean is None: return 0.0
    return round(float((1 - norm.cdf(thr, mean, max(std or 0.1, 0.1))) * 100), 1)

def gauss_below(mean, std, thr):
    if mean is None: return 0.0
    return round(float(norm.cdf(thr, mean, max(std or 0.1, 0.1)) * 100), 1)

def pct_to_gaussian(p10, p50, p90):
    if p50 is None: return None, None
    std = (p90 - p50) / 1.28 if (p90 and p90 > p50) else (p50 - p10) / 1.28 if (p10 and p50 > p10) else 3.0
    return p50, max(std, 0.5)

def _detect_nbm_format(section):
    """
    Identify NBM product layout from the section header.
    Returns (value_start_col, field_width, has_pipe_delim).
      NBH / NBS : labels in cols 1-4, values start col 5, width 3
      NBE / NBX / NBP : labels in cols 1-6, values start col 7, width 4, with | day separators
    """
    head = section[:600]
    if 'NBE GUIDANCE' in head or 'NBX GUIDANCE' in head or 'NBP GUIDANCE' in head:
        return 7, 4, True
    return 5, 3, False

_NBM_SENTINELS = {-99, 999, -88, 888}

def _nbm_to_int(s):
    if not s:
        return None
    try:
        v = int(s)
        return None if v in _NBM_SENTINELS else v
    except ValueError:
        return None

def parse_row(label, section, n_cols=None):
    """
    Parse one labeled row from an NBM bulletin section using fixed-width
    column chunking. Auto-detects NBH/NBS vs NBE/NBX/NBP format from header.

    Returns a list of ints/None, aligned to the UTC row's columns. If n_cols
    is supplied, the result is padded or truncated to exactly that length.
    Sentinels (-88, -99, 888, 999) become None. Blank slots (sparse rows like
    P06, T06, TXN) also become None — their valid-time alignment is preserved.
    """
    value_start, width, has_pipe = _detect_nbm_format(section)

    for line in section.split('\n'):
        if len(line) < value_start:
            continue
        prefix = line[:value_start]
        tokens = prefix.split()
        if not tokens or tokens[0] != label:
            continue

        data = line[value_start:]
        if has_pipe:
            data = data.replace('|', ' ')

        vals = [_nbm_to_int(data[i:i+width].strip())
                for i in range(0, len(data), width)]

        if n_cols is not None:
            if len(vals) < n_cols:
                vals = vals + [None] * (n_cols - len(vals))
            else:
                vals = vals[:n_cols]
        return vals

    return [None] * n_cols if n_cols is not None else []

def _utc_col_count(section):
    """Return the true UTC column count for this section (drops trailing-whitespace artifacts)."""
    utc = parse_row('UTC', section)
    while utc and utc[-1] is None:
        utc.pop()
    return len(utc), utc

def get_cycle(btype):
    now = datetime.now(timezone.utc)
    valid = [1, 7, 13, 19] if btype == 'p' else list(range(24))
    for hb in range(8):
        t = now - timedelta(hours=hb)
        if t.hour not in valid: continue
        ds, hs = t.strftime('%Y%m%d'), f"{t.hour:02d}"
        url = f"{NOMADS_NBM}/blend.{ds}/{hs}/text/blend_nb{btype}tx.t{hs}z"
        try:
            if requests.head(url, timeout=10).status_code == 200: return ds, hs, url
        except: continue
    return None, None, None

def fetch_station(url):
    r = requests.get(url, timeout=90)
    if r.status_code != 200: return None
    idx = r.text.find(STATION)
    if idx < 0: return None
    end = r.text.find('\n ' + STATION[:3], idx + 100)
    return r.text[idx: end if end > 0 else idx + 5000]

def parse_nbh(sec):
    n_cols, utc_row = _utc_col_count(sec)
    elements = ['TMP','TSD','DPT','DSD','WDR','WSP','GST','GSD','SKY',
                'P01','Q01','T01','VIS','MVV','IFV','LIV']
    rows = {el: parse_row(el, sec, n_cols=n_cols) for el in elements}
    data = {}
    # NBH column 0 is the cycle hour itself (fxx=0). Map fxx 1..N to columns 1..N.
    for i in range(1, min(n_cols, 49)):
        entry = {'utc_hour': utc_row[i], 'fxx': i}
        for el, vals in rows.items():
            entry[el] = vals[i]   # already None for sentinels/blanks
        data[i] = entry
    return data

def parse_nbp(sec):
    n_cols, _utc = _utc_col_count(sec)
    r = {}
    g24p1 = parse_row('G24P1', sec, n_cols=n_cols)
    g24p5 = parse_row('G24P5', sec, n_cols=n_cols)
    g24p9 = parse_row('G24P9', sec, n_cols=n_cols)
    txnp1 = parse_row('TXNP1', sec, n_cols=n_cols)
    txnp5 = parse_row('TXNP5', sec, n_cols=n_cols)
    txnp9 = parse_row('TXNP9', sec, n_cols=n_cols)
    if len(txnp5) >= 2 and txnp5[0] is not None:
        r.update({'TMAX_D1_P10': txnp1[0], 'TMAX_D1_P50': txnp5[0],
                  'TMAX_D1_P90': txnp9[0], 'TMIN_D1_P50': txnp5[1]})
    for pct, row in [(10, g24p1), (50, g24p5), (90, g24p9)]:
        if row and row[0] is not None:
            r[f'G24_D1_P{pct}'] = round(row[0] * KT_TO_MPH, 1)
    return r

def make_blocks(nbh, nbs):
    if not nbh.get(1): return [None]*16
    blocks = []
    for bi in range(16):
        s, e = bi * 3 + 1, bi * 3 + 3
        nbh_hrs = [nbh.get(f) for f in range(s, e + 1) if nbh.get(f)]
        if not nbh_hrs: blocks.append(None); continue
        def av(k):
            v = [h[k] for h in nbh_hrs if h.get(k) is not None]
            return sum(v)/len(v) if v else None
        blocks.append({
            'start_fxx': s, 'end_fxx': e, 'utc_start': nbh_hrs[0]['utc_hour'],
            'TMP': av('TMP'), 'TSD': av('TSD'), 'DPT': av('DPT'),
            'WDR': nbh_hrs[0].get('WDR'), 'WSP': round((av('WSP') or 0)*KT_TO_MPH, 1),
            'GST': round((av('GST') or 0)*KT_TO_MPH, 1), 'GSD': round((av('GSD') or 3)*KT_TO_MPH, 1),
            'SKY': av('SKY'), 'T01': max([h.get('T01') or 0 for h in nbh_hrs]),
            'P01': max([h.get('P01') or 0 for h in nbh_hrs]), 'Q01': av('Q01'),
            'VIS': min([h.get('VIS') or 100 for h in nbh_hrs]), 'LIV': max([h.get('LIV') or 0 for h in nbh_hrs])
        })
    return blocks

def compute_block(block, bi, nbp):
    if not block: return {hz: {"prob":0,"risk":0,"level":0,"color":RISK_C[0]} for hz in HAZARDS + ["COLD","HEAT"]}
    h = {}
    # WIND Logic Fix
    gst, gsd = block['GST'], block['GSD']
    best_p, best_l, best_rk = 0.0, 0, 0
    for t, l in [(65,5),(58,4),(45,3),(30,2)]:
        p = gauss_above(gst, gsd, t)
        rk = risk_matrix(p, l)
        if rk > best_rk or (rk == best_rk and p > best_p): best_p, best_l, best_rk = p, l, rk
    h["WIND"] = {"prob": best_p, "risk": best_rk, "level": best_l, "color": RISK_C[best_rk]}
    # LIGHTNING Fix
    t01 = float(block.get('T01') or 0)
    ll = 5 if t01>=75 else 4 if t01>=50 else 3 if t01>=25 else 2 if t01>=5 else 1 if t01>0 else 0
    h["LIGHTNING"] = {"prob": t01, "risk": risk_matrix(t01, ll), "level": ll, "color": RISK_C[risk_matrix(t01, ll)]}
    for hz in ["SNOW","VISIBILITY","FZRA","FLASH_FREEZE","RAIN"]: h[hz] = {"prob":0,"risk":0,"level":0,"color":RISK_C[0]}
    # TEMPERATURE Fix
    tmp, tsd = block['TMP'], block['TSD'] or 3
    cp, cl, hp, hl = 0, 0, 0, 0
    for t, l in [(40,2),(32,3),(20,4),(10,5)]:
        p = gauss_below(tmp, tsd, t)
        if p >= 5.0: cp, cl = p, l
    for t, l in [(85,1),(90,2),(95,3),(100,4),(105,5)]:
        p = gauss_above(tmp, tsd, t)
        if p >= 5.0: hp, hl = p, l
    if risk_matrix(hp, hl) >= risk_matrix(cp, cl):
        h["TEMPERATURE"] = {"prob": hp, "risk": risk_matrix(hp, hl), "level": hl, "color": RISK_C[risk_matrix(hp, hl)], "temp_type": "heat"}
    else:
        h["TEMPERATURE"] = {"prob": cp, "risk": risk_matrix(cp, cl), "level": cl, "color": RISK_C[risk_matrix(cp, cl)], "temp_type": "cold"}
    return h

def main():
    ds, hs, url = get_cycle('h')
    nbh_sec = fetch_station(url)
    nbp_sec = fetch_station(get_cycle('p')[2])
    if not nbh_sec: sys.exit(1)
    nbh, nbp = parse_nbh(nbh_sec), parse_nbp(nbp_sec) if nbp_sec else {}
    blocks = make_blocks(nbh, {})
    block_hazards = [compute_block(b, i, nbp) for i, b in enumerate(blocks)]
    threats = {}
    for hz in HAZARDS:
        idx = [i for i in range(len(blocks)) if blocks[i]]
        pk = max(idx, key=lambda i: (block_hazards[i][hz]["risk"], block_hazards[i][hz]["prob"]))
        threats[hz] = {
            "prob": block_hazards[pk][hz]["prob"], "risk": block_hazards[pk][hz]["risk"], 
            "risk_label": RISK_L[block_hazards[pk][hz]["risk"]], "color": block_hazards[pk][hz]["color"], 
            "level": block_hazards[pk][hz]["level"], "metric": METRICS.get(hz, {}).get(block_hazards[pk][hz]["level"], ""),
            "peak_start_fxx": blocks[pk]["start_fxx"], "peak_end_fxx": blocks[pk]["end_fxx"], "peak_utc_start": blocks[pk]["utc_start"]
        }
    # WIND OVERRIDE WITH P10/P50/P90
    p10, p50, p90 = nbp.get('G24_D1_P10'), nbp.get('G24_D1_P50'), nbp.get('G24_D1_P90')
    if p50:
        gm, gs = pct_to_gaussian(p10, p50, p90)
        best_p, best_l, best_rk = 0, 0, 0
        for t, l in [(65,5), (58,4), (45,3), (30,2)]:
            p = gauss_above(gm, gs, t)
            rk = risk_matrix(p, l)
            if rk > best_rk or (rk == best_rk and p > best_p): best_p, best_l, best_rk = p, l, rk
        if best_rk >= threats['WIND']['risk']:
            threats['WIND'].update({
                "prob": best_p, "risk": best_rk, "risk_label": RISK_L[best_rk], "color": RISK_C[best_rk], 
                "level": best_l, "metric": METRICS["WIND"].get(best_l, ""), 
                "g24_p10_mph": p10, "g24_p50_mph": p50, "g24_p90_mph": p90
            })

    nbh_hourly = []
    for fxx in range(1, 26):
        h = nbh.get(fxx, {})
        def _kt(v): return round((v or 0)*KT_TO_MPH, 1)
        nbh_hourly.append({
            'fxx': fxx, 'utc': h.get('utc_hour'), 'TMP': h.get('TMP'), 'TSD': h.get('TSD'), 
            'DPT': h.get('DPT'), 'WDR': h.get('WDR'), 'WSP': _kt(h.get('WSP')), 'GST': _kt(h.get('GST')), 
            'GSD': _kt(h.get('GSD')), 'SKY': h.get('SKY'), 'T01': h.get('T01'), 'P01': h.get('P01'), 
            'Q01': h.get('Q01'), 'VIS': h.get('VIS'), 'LIV': h.get('LIV'), 'IFV': h.get('IFV'), 'MVV': h.get('MVV')
        })
    with open('threats.json', 'w') as f: json.dump({"threats": threats, "cycle_utc_iso": f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}T{hs}:00:00Z"}, f)
    with open('timeline.json', 'w') as f: json.dump({"blocks": blocks, "block_hazards": block_hazards, "nbh_hourly": nbh_hourly}, f)

if __name__ == "__main__": main()
