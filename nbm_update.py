#!/usr/bin/env python3
"""
KRNO DSS — NBM Text Bulletin Processor
GitHub Actions workflow script
Downloads NBM text bulletins, computes hazard probabilities,
writes threats.json and timeline.json to repo root.

DATA SOURCES:
  NBH (hourly)  : f001-f025   1-hr resolution
  NBS (3-hourly): f005-f072   3-hr resolution; fills blocks 9-15 (hrs 26-48)
  NBP (prob)    : 01/07/13/19Z only -- G24 gust percentiles, MaxT/MinT
"""

import re, json, requests, sys
from datetime import datetime, timezone, timedelta
from scipy.stats import norm
import numpy as np

KT_TO_MPH = 1.15078  # nautical miles per hour → statute mph

STATION    = "KRNO"
NOMADS_NBM = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"

# -- DESI 2D Risk Matrix -------------------------------------------------------
# Rows = probability bands (0=<10%, 1=10-33%, 2=33-66%, 3=66-90%, 4=>90%)
# Cols = impact level 1-5
# Corrected to match the official NWS likelihood x impact matrix 
_MATRIX = {
    4: [1, 2, 3, 4, 5],   # >90%  Very Likely
    3: [1, 2, 3, 4, 4],   # >66%  Likely
    2: [1, 2, 2, 3, 4],   # >33%  As Likely As Not
    1: [1, 1, 2, 2, 3],   # >10%  Unlikely
    0: [1, 1, 1, 2, 2],   # <10%  Extremely Unlikely
}

def risk_matrix(prob, level):
    """Two-dimensional risk lookup: probability x impact level -> risk (0-5)."""
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

# -- Math helpers --------------------------------------------------------------

def gauss_above(mean, std, thr):
    if mean is None: return 0.0
    return round(float((1 - norm.cdf(thr, mean, max(std or 0.1, 0.1))) * 100), 1)

def gauss_below(mean, std, thr):
    if mean is None: return 0.0
    return round(float(norm.cdf(thr, mean, max(std or 0.1, 0.1)) * 100), 1)

def pct_to_gaussian(p10, p50, p90):
    if p50 is None: return None, None
    if p90 and p90 > p50: std = (p90 - p50) / 1.28
    elif p10 and p50 > p10: std = (p50 - p10) / 1.28
    else: std = 3.0
    return p50, max(std, 0.5)

def parse_row(label, section):
    for line in section.split('\n'):
        s = line.strip()
        if s.startswith(label + ' ') or s.startswith(label + '\t'):
            return [int(x) for x in re.findall(r'-?\d+', s[len(label):])]
    return []

# -- Fetch ---------------------------------------------------------------------

def get_cycle(btype):
    now = datetime.now(timezone.utc)
    valid = [1, 7, 13, 19] if btype == 'p' else list(range(24))
    for hb in range(8):
        t = now - timedelta(hours=hb)
        if t.hour not in valid: continue
        ds, hs = t.strftime('%Y%m%d'), f"{t.hour:02d}"
        url = f"{NOMADS_NBM}/blend.{ds}/{hs}/text/blend_nb{btype}tx.t{hs}z"
        try:
            if requests.head(url, timeout=10).status_code == 200:
                return ds, hs, url
        except: continue
    return None, None, None

def fetch_station(url):
    r = requests.get(url, timeout=90)
    if r.status_code != 200: return None
    idx = r.text.find(STATION)
    if idx < 0: return None
    end = r.text.find('\n ' + STATION[:3], idx + 100)
    return r.text[idx: end if end > 0 else idx + 5000]

# -- Parse bulletins -----------------------------------------------------------

def parse_nbh(sec):
    utc_row = []
    for line in sec.split('\n'):
        if line.strip().startswith('UTC '):
            utc_row = [int(x) for x in re.findall(r'\d+', line[4:])]
            break
    rows = {}
    for el in ['TMP','TSD','DPT','DSD','WDR','WSP','GST','GSD','SKY','P01','Q01','T01','VIS','MVV','IFV','LIV']:
        v = parse_row(el, sec)
        if v: rows[el] = v
    data = {}
    for i, uh in enumerate(utc_row[:48]):
        fxx = i + 1
        entry = {'utc_hour': uh, 'fxx': fxx}
        for el, vals in rows.items():
            if i < len(vals):
                v = vals[i]
                entry[el] = None if v in (-99, 999, -88, 888) else v
        data[fxx] = entry
    return data

def parse_nbs(sec):
    fhr_row = []
    for line in sec.split('\n'):
        if 'FHR' in line:
            fhr_row = [int(x) for x in re.findall(r'\d+', line[4:])]
            break
    rows = {}
    for el in ['TMP','TSD','DPT','DSD','WDR','WSP','GST','GSD','SKY','P06','Q06','S06','ZR6','T06']:
        v = parse_row(el, sec)
        if v: rows[el] = v
    data = {}
    for i, fxx in enumerate(fhr_row):
        if fxx > 72: break
        entry = {'fxx': fxx}
        for el, vals in rows.items():
            if i < len(vals):
                v = vals[i]
                entry[el] = None if v in (-99, 999) else v
        data[fxx] = entry
    return data

def parse_nbp(sec):
    g24p1 = parse_row('G24P1', sec); g24p2 = parse_row('G24P2', sec)
    g24p5 = parse_row('G24P5', sec); g24p7 = parse_row('G24P7', sec)
    g24p9 = parse_row('G24P9', sec)
    txnp1 = parse_row('TXNP1', sec)
    txnp5 = parse_row('TXNP5', sec)
    txnp9 = parse_row('TXNP9', sec)
    r = {}
    if len(txnp5) >= 2:
        r.update({'TMAX_D1_P10': min(txnp1[0], txnp1[1]) if len(txnp1)>=2 else None,
                  'TMAX_D1_P50': max(txnp5[0], txnp5[1]),
                  'TMAX_D1_P90': max(txnp9[0], txnp9[1]) if len(txnp9)>=2 else None,
                  'TMIN_D1_P10': min(txnp1[0], txnp1[1]) if len(txnp1)>=2 else None,
                  'TMIN_D1_P50': min(txnp5[0], txnp5[1]),
                  'TMIN_D1_P90': max(txnp9[0], txnp9[1]) if len(txnp9)>=2 else None})
    if len(txnp5) >= 4:
        r.update({'TMAX_D2_P10': min(txnp1[2], txnp1[3]) if len(txnp1)>=4 else None,
                  'TMAX_D2_P50': max(txnp5[2], txnp5[3]),
                  'TMAX_D2_P90': max(txnp9[2], txnp9[3]) if len(txnp9)>=4 else None,
                  'TMIN_D2_P50': min(txnp5[2], txnp5[3])})
    for pct, row in [(10,g24p1),(20,g24p2),(50,g24p5),(70,g24p7),(90,g24p9)]:
        if len(row) >= 1: r[f'G24_D1_P{pct}'] = round(row[0] * KT_TO_MPH, 1)
        if len(row) >= 2: r[f'G24_D2_P{pct}'] = round(row[1] * KT_TO_MPH, 1)
    return r

def make_blocks(nbh, nbs):
    cycle_utc_h = None
    if nbh.get(1):
        cycle_utc_h = (nbh[1]['utc_hour'] - 1) % 24
    nbs_keys = sorted(nbs.keys()) if nbs else []
    def closest_nbs(mid):
        if not nbs_keys: return None
        ck = min(nbs_keys, key=lambda k: abs(k - mid))
        return nbs[ck] if abs(ck - mid) <= 5 else None
    blocks = []
    for bi in range(16):
        s, e = bi * 3 + 1, bi * 3 + 3
        mid = s + 1
        nbh_hrs = [nbh.get(f) for f in range(s, e + 1) if nbh.get(f)]
        nbs_entry = closest_nbs(mid)
        if not nbh_hrs and nbs_entry is None:
            blocks.append(None); continue
        def _nbs(k): return nbs_entry.get(k) if nbs_entry else None
        def av(k):
            v = [h[k] for h in nbh_hrs if h.get(k) is not None]
            return sum(v) / len(v) if v else _nbs(k)
        def mx(k):
            v = [h[k] for h in nbh_hrs if h.get(k) is not None]
            return max(v) if v else _nbs(k)
        def mn(k):
            v = [h[k] for h in nbh_hrs if h.get(k) is not None]
            return min(v) if v else _nbs(k)
        utc_start = nbh_hrs[0].get('utc_hour') if nbh_hrs else (cycle_utc_h + s) % 24 if cycle_utc_h is not None else None
        b = {
            'start_fxx': s, 'end_fxx': e, 'utc_start': utc_start,
            'TMP': av('TMP'), 'TSD': av('TSD'), 'DPT': av('DPT'), 'WDR': mx('WDR'),
            'WSP': round(mx('WSP') * KT_TO_MPH, 1), 'GST': round(mx('GST') * KT_TO_MPH, 1),
            'GSD': round(av('GSD') * KT_TO_MPH, 1), 'SKY': av('SKY'),
            'T01': mx('T01') if nbh_hrs else _nbs('T06'),
            'P01': mx('P01') if nbh_hrs else _nbs('P06'),
            'Q01': mx('Q01') if nbh_hrs else round(_nbs('Q06')/6.0, 1) if _nbs('Q06') else None,
            'VIS': mn('VIS'), 'MVV': mx('MVV'), 'IFV': mx('IFV'), 'LIV': mx('LIV'),
            'S06': _nbs('S06'), 'ZR6': _nbs('ZR6'), 'T06': _nbs('T06'),
        }
        blocks.append(b)
    return blocks

# -- Compute block hazards -----------------------------------------------------

def compute_block(block, bi, nbp):
    if block is None:
        return {hz: {"prob":0,"risk":0,"level":0,"color":RISK_C[0]}
                for hz in HAZARDS + ["COLD","HEAT"]}
    h = {}
    d2 = bi >= 8
    
    # -- WIND (time-varying) --------------------------------------------------
    gst = block.get('GST')
    gsd = max(block.get('GSD') or 3, 2.0)
    wp, wl = 0.0, 0
    if gst and gst >= 10:
        for t, l in [(65,5),(58,4),(45,3),(30,2)]:
            p = gauss_above(gst, gsd, t)
            if risk_matrix(p, l) >= risk_matrix(wp, wl): wp, wl = p, l
    rk = risk_matrix(wp, wl)
    h["WIND"] = {"prob": wp, "risk": rk, "level": wl, "color": RISK_C[rk]}

    # -- LIGHTNING ------------------------------------------------------------
    t01 = block.get('T06') if d2 else block.get('T01') or 0
    ll = 5 if t01>=75 else 4 if t01>=50 else 3 if t01>=25 else 2 if t01>=5 else 1 if t01>0 else 0
    h["LIGHTNING"] = {"prob": float(t01), "risk": risk_matrix(t01, ll), "level": ll, "color": RISK_C[risk_matrix(t01, ll)]}

    # -- SNOW -----------------------------------------------------------------
    s06 = block.get('S06'); pop = block.get('P01') or 0; sp, sl = 0, 0
    if s06 and s06 > 0:
        s_inhr = (s06 / 10.0) / 6.0
        for t, l in [(2,5),(1,4),(0.5,3),(0.1,2)]:
            if s_inhr >= t: sp, sl = min(pop, 100.0), l; break
    h["SNOW"] = {"prob": sp, "risk": risk_matrix(sp, sl), "level": sl, "color": RISK_C[risk_matrix(sp, sl)]}

    # -- VISIBILITY -----------------------------------------------------------
    liv, ifv, mvv = block.get('LIV') or 0, block.get('IFV') or 0, block.get('MVV') or 0
    if liv >= 20: vp, vl = liv, 4 # Increased threshold to 20% to avoid false alarms
    elif ifv >= 25: vp, vl = ifv, 3
    elif mvv >= 33: vp, vl = mvv, 2
    else: vp, vl = 0, 0
    h["VISIBILITY"] = {"prob": vp, "risk": risk_matrix(vp, vl), "level": vl, "color": RISK_C[risk_matrix(vp, vl)]}

    # -- FZRA -----------------------------------------------------------------
    zr6 = block.get('ZR6'); fzp, fzl = 0, 0
    if zr6 and zr6 > 0:
        zr = zr6 / 100.0
        for t, l in [(0.10,5),(0.01,4),(0.001,3),(0.0001,2)]:
            if zr >= t: fzp, fzl = min(pop, 100.0), l; break
    h["FZRA"] = {"prob": fzp, "risk": risk_matrix(fzp, fzl), "level": fzl, "color": RISK_C[risk_matrix(fzp, fzl)]}

    # -- FLASH FREEZE ---------------------------------------------------------
    tmp_f, dpt_f, pop = block.get('TMP'), block.get('DPT'), block.get('P01') or 0
    ff_p, ff_l = 0, 0
    if tmp_f is not None and dpt_f is not None and pop >= 25:
        tw_f = tmp_f - (tmp_f - dpt_f) / 3.0
        for t, l in [(25,5),(28,4),(32,3),(36,2)]:
            if tw_f <= t: ff_p, ff_l = pop, l; break
    h["FLASH_FREEZE"] = {"prob": ff_p, "risk": risk_matrix(ff_p, ff_l), "level": ff_l, "color": RISK_C[risk_matrix(ff_p, ff_l)]}

    # -- RAIN -----------------------------------------------------------------
    q01 = block.get('Q01'); rp, rl = 0, 0
    if q01 and q01 > 0:
        q = q01 / 100.0
        for t, l in [(1.0,5),(0.5,4),(0.25,3),(0.10,2)]:
            if q >= t: rp, rl = min(pop, 100.0), l; break
    h["RAIN"] = {"prob": rp, "risk": risk_matrix(rp, rl), "level": rl, "color": RISK_C[risk_matrix(rp, rl)]}

    # -- TEMPERATURE (COLD/HEAT) ----------------------------------------------
    tmp, tsd = block.get('TMP'), max(block.get('TSD') or 3, 0.5)
    cp, cl, hp, hl = 0.0, 0, 0.0, 0
    if tmp is not None:
        for t, l in [(40,2),(32,3),(20,4),(10,5)]: # Corrected Cold Impact Level 5 to <10F
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

# -- Main ----------------------------------------------------------------------

def main():
    print("KRNO DSS -- NBM processor (v4: Scoped logic fixes)")
    nbh_ds, nbh_hs, nbh_url = get_cycle('h')
    _, nbs_hs, nbs_url = get_cycle('s')
    _, nbp_hs, nbp_url = get_cycle('p')
    nbh_sec = fetch_station(nbh_url) if nbh_url else None
    nbs_sec = fetch_station(nbs_url) if nbs_url else None
    nbp_sec = fetch_station(nbp_url) if nbp_url else None
    if not nbh_sec: sys.exit(1)
    nbh, nbs, nbp = parse_nbh(nbh_sec), parse_nbs(nbs_sec) if nbs_sec else {}, parse_nbp(nbp_sec) if nbp_sec else {}
    blocks = make_blocks(nbh, nbs)
    block_hazards = [compute_block(b, i, nbp) for i, b in enumerate(blocks)]
    threats = {}
    for hz in HAZARDS:
        idx = [i for i in range(16) if blocks[i]]
        if not idx: continue
        pk = max(idx, key=lambda i: (block_hazards[i][hz]["risk"], block_hazards[i][hz]["prob"]))
        hdata = block_hazards[pk][hz]
        met = METRICS.get(hz, {}).get(hdata["level"], "") if hdata["level"] > 0 else ""
        threats[hz] = {
            "prob": round(hdata["prob"], 1), "risk": hdata["risk"], "risk_label": RISK_L[hdata["risk"]],
            "color": hdata["color"], "level": hdata["level"], "metric": met,
            "peak_start_fxx": blocks[pk]["start_fxx"], "peak_end_fxx": blocks[pk]["end_fxx"], "peak_utc_start": blocks[pk]["utc_start"]
        }
        
    # -- Card Overrides (WIND/TEMP via NBP Distribution) ----------------------
    for day, bi_range in [('D1', range(8)), ('D2', range(8, 16))]:
        p10, p50, p90 = nbp.get(f'G24_{day}_P10'), nbp.get(f'G24_{day}_P50'), nbp.get(f'G24_{day}_P90')
        if p50:
            gm, gs = pct_to_gaussian(p10, p50, p90)
            wp, wl = 0.0, 0
            for t, l in [(65,5), (58,4), (45,3), (30,2)]: # Sweeping all levels for max risk
                p = gauss_above(gm, gs, t)
                if risk_matrix(p, l) >= risk_matrix(wp, wl): wp, wl = p, l
            rk = risk_matrix(wp, wl)
            if rk >= threats['WIND']['risk']:
                threats['WIND'].update({"prob": wp, "risk": rk, "risk_label": RISK_L[rk], "color": RISK_C[rk], "level": wl, "metric": METRICS["WIND"].get(wl, ""), "g24_p50_mph": p50, "g24_p90_mph": p90})

    cycle_utc_iso = f"{nbh_ds[:4]}-{nbh_ds[4:6]}-{nbh_ds[6:8]}T{nbh_hs}:00:00Z" if nbh_ds else None
    with open('threats.json', 'w') as f: json.dump({"threats": threats, "cycle_utc_iso": cycle_utc_iso}, f)
    with open('timeline.json', 'w') as f: json.dump({"blocks": blocks, "block_hazards": block_hazards}, f)
    print("Processing Complete.")

if __name__ == "__main__": main()
