#!/usr/bin/env python3
"""
KRNO DSS — NBM Text Bulletin Processor
GitHub Actions workflow script
Downloads NBM text bulletins, computes hazard probabilities,
writes threats.json and timeline.json to repo root.

DATA SOURCES:
  NBH (hourly)  : f001-f025   1-hr resolution (25 hrs by product design)
  NBS (3-hourly): f005-f072   3-hr resolution; fills blocks 9-15 (hrs 26-48)
  NBP (prob)    : 01/07/13/19Z only -- G24 gust percentiles, MaxT/MinT

RISK METHODOLOGY:
  Two-dimensional DESI matrix: probability row x impact level column.
  Risk = matrix[prob_band][impact_level], not probability alone.
  This matches the NWS likelihood x consequence framework.

  WIND timeline: uses block GST/GSD (time-varying hourly signal from NBH/NBS)
  WIND card:     max of timeline blocks (peak period risk)
  G24 from NBP:  printed in log for reference/verification only
"""

import re, json, requests, sys
from datetime import datetime, timezone, timedelta
from scipy.stats import norm
import numpy as np

STATION    = "KRNO"
NOMADS_NBM = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"

# -- DESI 2D Risk Matrix -------------------------------------------------------
# Rows = probability bands (0=<10%, 1=10-33%, 2=33-66%, 3=66-90%, 4=>90%)
# Cols = impact level 1-5
# Values match the NWS likelihood x impact color matrix (Image 1).
_MATRIX = {
    4: [1, 2, 3, 4, 5],   # >90%  Very Likely
    3: [1, 2, 3, 4, 4],   # >66%  Likely
    2: [1, 2, 2, 3, 4],   # >33%  As Likely As Not
    1: [1, 2, 2, 2, 3],   # >10%  Unlikely
    0: [1, 2, 2, 2, 3],   # <10%  Extremely Unlikely
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
    "HEAT":        {2:"90-95F",       3:"95-100F",      4:"100-105F",    5:">105F"},
    "COLD":        {2:"32-40F",       3:"20-32F",       4:"0-20F",       5:"<0F"},
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
    """NBH: hourly, f001-f025 by product design."""
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
    """NBS: 3-hourly, f005-f072. Fills met elements for hours 26-48."""
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
    """
    NBP probabilistic bulletin.
    G24P values are in KNOTS -- converted to mph on storage.
    TXNMN column order is cycle-dependent; use min/max of each pair.
    """
    g24p1 = parse_row('G24P1', sec); g24p2 = parse_row('G24P2', sec)
    g24p5 = parse_row('G24P5', sec); g24p7 = parse_row('G24P7', sec)
    g24p9 = parse_row('G24P9', sec)
    # TXNP percentile rows -- used for temperature Gaussian (more accurate than TXNMN/TXNSD)
    # TXNP1=10th pctile, TXNP5=50th pctile (median), TXNP9=90th pctile
    # Col order in NBP: alternates MinT/MaxT per 12z period; same cycle-dependency as TXNMN.
    # Use max()=MaxT, min()=MinT within each pair -- robust to cycle ordering.
    txnp1 = parse_row('TXNP1', sec)  # 10th percentile
    txnp5 = parse_row('TXNP5', sec)  # 50th percentile (median)
    txnp9 = parse_row('TXNP9', sec)  # 90th percentile
    r = {}
    # Temperature: prefer TXNP percentile approach (matches histogram values exactly).
    # Falls back to TXNMN/TXNSD if TXNP rows are absent.
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
    # Fallback: TXNMN/TXNSD if no TXNP data
    if 'TMAX_D1_P50' not in r:
        txnmn = parse_row('TXNMN', sec); txnsd = parse_row('TXNSD', sec)
        if len(txnmn) >= 2:
            r.update({'TMAX_D1_P50': max(txnmn[0], txnmn[1]),
                      'TMIN_D1_P50': min(txnmn[0], txnmn[1])})
        if len(txnmn) >= 4:
            r.update({'TMAX_D2_P50': max(txnmn[2], txnmn[3]),
                      'TMIN_D2_P50': min(txnmn[2], txnmn[3])})
        if len(txnsd) >= 2:
            r.update({'TMAX_D1_STD': max(txnsd[0], txnsd[1]),
                      'TMIN_D1_STD': min(txnsd[0], txnsd[1])})
        if len(txnsd) >= 4:
            r.update({'TMAX_D2_STD': max(txnsd[2], txnsd[3]),
                      'TMIN_D2_STD': min(txnsd[2], txnsd[3])})
    # G24P in KNOTS -- convert to mph
    KT_TO_MPH = 1.15078
    for pct, row in [(10,g24p1),(20,g24p2),(50,g24p5),(70,g24p7),(90,g24p9)]:
        if len(row) >= 1: r[f'G24_D1_P{pct}'] = round(row[0] * KT_TO_MPH, 1)
        if len(row) >= 2: r[f'G24_D2_P{pct}'] = round(row[1] * KT_TO_MPH, 1)
    return r

# -- Build 3-hour blocks -------------------------------------------------------

def make_blocks(nbh, nbs):
    """
    16 x 3-hour blocks covering f001-f048.
      Blocks 0-7  (fxx 1-24):  NBH primary
      Block  8    (fxx 25-27): NBH fxx=25 + NBS
      Blocks 9-15 (fxx 28-48): NBS primary (VIS fields unavailable)

    NBS LOOKUP: uses closest FHR key (fixes the round(mid/3)*3 arithmetic
    that was missing every NBS key and leaving all D2 blocks empty).
    """
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
        s = bi * 3 + 1
        e = s + 2
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

        if nbh_hrs:
            utc_start = nbh_hrs[0].get('utc_hour')
        elif cycle_utc_h is not None:
            utc_start = (cycle_utc_h + s) % 24
        else:
            utc_start = None

        nbs_p01 = _nbs('P06')
        nbs_q01 = round(_nbs('Q06') / 6.0, 1) if _nbs('Q06') is not None else None

        b = {
            'start_fxx': s, 'end_fxx': e, 'utc_start': utc_start,
            'TMP': av('TMP'), 'TSD': av('TSD'), 'DPT': av('DPT'),
            'WDR': mx('WDR'),
            # WSP/GST/GSD from NBH/NBS are in KNOTS — convert to mph for all downstream use
            'WSP': round(mx('WSP') * KT_TO_MPH, 1),
            'GST': round(mx('GST') * KT_TO_MPH, 1),
            'GSD': round(av('GSD') * KT_TO_MPH, 1),
            'SKY': av('SKY'),  # cloud coverage %
            'T01': mx('T01') if nbh_hrs else _nbs('T06'),
            'P01': mx('P01') if nbh_hrs else nbs_p01,
            'Q01': mx('Q01') if nbh_hrs else nbs_q01,
            'VIS': mn('VIS'), 'MVV': mx('MVV'), 'IFV': mx('IFV'), 'LIV': mx('LIV'),
            'S06': _nbs('S06'), 'ZR6': _nbs('ZR6'), 'T06': _nbs('T06'),
            # P90 approx for tooltip display: mean + 1.28 * std
            'GST_P90': round(mx('GST') + 1.28 * (av('GSD') or 3), 1) if (mx('GST') and av('GSD')) else mx('GST'),
            'TMP_P90': round(av('TMP') + 1.28 * (av('TSD') or 3), 1) if (av('TMP') and av('TSD')) else av('TMP'),
            'TMP_P10': round(av('TMP') - 1.28 * (av('TSD') or 3), 1) if (av('TMP') and av('TSD')) else av('TMP'),
        }
        blocks.append(b)
    return blocks

# -- Compute block hazards -----------------------------------------------------

def compute_block(block, bi, nbp):
    """
    Compute hazard risk for one 3-hour block.

    WIND uses block GST/GSD for a time-varying signal (not G24 24-hr max).
    All risks use the 2D probability x impact matrix, not probability alone.
    """
    if block is None:
        return {hz: {"prob":0,"risk":0,"level":0,"color":RISK_C[0]}
                for hz in HAZARDS + ["COLD","HEAT"]}
    h = {}
    d2 = bi >= 8

    # -- WIND (block GST -- time-varying) -------------------------------------
    gst = block.get('GST')
    gsd = max(block.get('GSD') or 3, 2.0)
    wp, wl = 0.0, 0
    if gst and gst >= 10:
        if   gst >= 65: wl, low = 5, 65
        elif gst >= 58: wl, low = 4, 58
        elif gst >= 45: wl, low = 3, 45
        elif gst >= 30: wl, low = 2, 30
        else:           wl, low = 0, 30
        wp = gauss_above(gst, gsd, low)
        if wl == 0 and wp < 5.0: wp, wl = 0.0, 0
        elif wl == 0: wl = 2
    rk = risk_matrix(wp, wl)
    h["WIND"] = {"prob": wp, "risk": rk, "level": wl, "color": RISK_C[rk]}

    # -- LIGHTNING ------------------------------------------------------------
    # D1 blocks: T01 only (NBH hourly — correctly attributed to the 1-hr slot).
    # D2 blocks: T06 from NBS (T01 unavailable; unavoidable 6-hr aggregate).
    # Never mix T01/T06 in D1: NBS T06 covers the NEXT 6 hours from that FHR,
    # which is an entirely different window than the 3-hr block it gets assigned to.
    if not d2:
        t01 = block.get('T01') or 0
    else:
        t01 = block.get('T06') or block.get('T01') or 0
    ll = 5 if t01>=75 else 4 if t01>=50 else 3 if t01>=25 else 2 if t01>=5 else 1 if t01>0 else 0
    h["LIGHTNING"] = {"prob": float(t01), "risk": risk_matrix(t01, ll),
                      "level": ll, "color": RISK_C[risk_matrix(t01, ll)]}

    # -- SNOW -----------------------------------------------------------------
    s06 = block.get('S06'); pop = block.get('P01') or 0; sp, sl = 0, 0
    if s06 and s06 > 0:
        s_inhr = (s06 / 10.0) / 6.0
        for t, l in [(2,5),(1,4),(0.5,3),(0.1,2)]:
            if s_inhr >= t: sp, sl = min(pop, 100.0), l; break
    h["SNOW"] = {"prob": sp, "risk": risk_matrix(sp, sl), "level": sl,
                 "color": RISK_C[risk_matrix(sp, sl)]}

    # -- VISIBILITY -----------------------------------------------------------
    liv = block.get('LIV') or 0; ifv = block.get('IFV') or 0; mvv = block.get('MVV') or 0
    # Require >=5% probability before registering a visibility hazard.
    # MVV/IFV/LIV=1 (single percent) generates MINOR via 2D matrix but is
    # operationally meaningless at that probability — suppress it.
    if liv >= 5: vp, vl = liv, 4
    elif ifv >= 5: vp, vl = ifv, 3
    elif mvv >= 5: vp, vl = mvv, 2
    else: vp, vl = 0, 0
    h["VISIBILITY"] = {"prob": vp, "risk": risk_matrix(vp, vl), "level": vl,
                       "color": RISK_C[risk_matrix(vp, vl)]}

    # -- FZRA -----------------------------------------------------------------
    zr6 = block.get('ZR6'); fzp, fzl = 0, 0
    if zr6 and zr6 > 0:
        zr = zr6 / 100.0
        for t, l in [(0.10,5),(0.01,4),(0.001,3),(0.0001,2)]:
            if zr >= t: fzp, fzl = min(pop, 100.0), l; break
    h["FZRA"] = {"prob": fzp, "risk": risk_matrix(fzp, fzl), "level": fzl,
                 "color": RISK_C[risk_matrix(fzp, fzl)]}

    # -- FLASH FREEZE ---------------------------------------------------------
    tmp_f = block.get('TMP'); dpt_f = block.get('DPT'); ff_p, ff_l = 0, 0
    if tmp_f is not None and dpt_f is not None and pop >= 25:
        tw_f = tmp_f - (tmp_f - dpt_f) / 3.0
        wf = pop / 100.0
        for t, l in [(25,5),(28,4),(32,3),(36,2)]:
            if tw_f <= t: ff_p = round(wf * 100, 1); ff_l = l; break
    h["FLASH_FREEZE"] = {"prob": ff_p, "risk": risk_matrix(ff_p, ff_l), "level": ff_l,
                         "color": RISK_C[risk_matrix(ff_p, ff_l)]}

    # -- RAIN -----------------------------------------------------------------
    q01 = block.get('Q01'); rp, rl = 0, 0
    if q01 and q01 > 0:
        q = q01 / 100.0
        for t, l in [(1.0,5),(0.5,4),(0.25,3),(0.10,2)]:
            if q >= t: rp, rl = min(pop, 100.0), l; break
    elif pop > 10: rp, rl = pop, 2
    h["RAIN"] = {"prob": rp, "risk": risk_matrix(rp, rl), "level": rl,
                 "color": RISK_C[risk_matrix(rp, rl)]}

    # -- COLD (block TMP -- time-varying) ------------------------------------
    # Use the block's actual forecast temperature so each 3-hr period reflects
    # when it's truly cold (overnight low), not a uniform daily min applied
    # to all 8 D1 blocks including the afternoon.
    tmp  = block.get('TMP')
    tsd  = max(block.get('TSD') or 3, 0.5)
    cp, cl = 0.0, 0
    if tmp is not None:
        for t, l in [(40,2),(32,3),(20,4),(0,5)]:
            p = gauss_below(tmp, tsd, t)
            if p >= 5.0: cp, cl = p, l
            else: break
    h["COLD"] = {"prob": cp, "risk": risk_matrix(cp, cl), "level": cl,
                 "color": RISK_C[risk_matrix(cp, cl)]}

    # -- HEAT (block TMP -- time-varying) ------------------------------------
    # Same approach as COLD: use block TMP so heat risk only appears during
    # the actual warm hours of the day. A 91F afternoon block triggers heat;
    # a 65F midnight block correctly shows nothing.
    hp, hl = 0.0, 0
    if tmp is not None:
        for t, l in [(90,2),(95,3),(100,4),(105,5)]:
            p = gauss_above(tmp, tsd, t)
            if p >= 5.0: hp, hl = p, l
            else: break
    h["HEAT"] = {"prob": hp, "risk": risk_matrix(hp, hl), "level": hl,
                 "color": RISK_C[risk_matrix(hp, hl)]}

    # -- TEMPERATURE (higher risk of COLD or HEAT) ----------------------------
    if h["HEAT"]["risk"] >= h["COLD"]["risk"]:
        temp = dict(h["HEAT"]); temp["temp_type"] = "heat"
    else:
        temp = dict(h["COLD"]); temp["temp_type"] = "cold"
    h["TEMPERATURE"] = temp
    # Remove internal COLD/HEAT — only TEMPERATURE appears in output.
    # Frontend timeline and cards should show TEMPERATURE row only (not separate cold/heat).
    del h["COLD"]; del h["HEAT"]

    return h

# -- Main ----------------------------------------------------------------------

def main():
    print("KRNO DSS -- NBM processor  (v3: 2D matrix, GST wind, NBS closest-key)")
    now = datetime.now(timezone.utc)
    print(f"Run time: {now.isoformat()}")

    print("\nFetching NBM bulletins...")
    nbh_ds, nbh_hs, nbh_url = get_cycle('h')
    _,       nbs_hs, nbs_url = get_cycle('s')
    _,       nbp_hs, nbp_url = get_cycle('p')
    print(f"  NBH: {nbh_hs}Z  NBS: {nbs_hs}Z  NBP: {nbp_hs}Z")

    nbh_sec = fetch_station(nbh_url) if nbh_url else None
    nbs_sec = fetch_station(nbs_url) if nbs_url else None
    nbp_sec = fetch_station(nbp_url) if nbp_url else None

    if not nbh_sec:
        print("ERROR: NBH bulletin unavailable", file=sys.stderr); sys.exit(1)

    nbh = parse_nbh(nbh_sec)
    nbs = parse_nbs(nbs_sec) if nbs_sec else {}
    nbp = parse_nbp(nbp_sec) if nbp_sec else {}

    print(f"  NBH: {len(nbh)} hrs  NBS: {len(nbs)} periods  NBP: {len(nbp)} fields")

    # Always log G24 values for cross-checking against NBM bulletin/histogram
    if nbp:
        for day in ['D1', 'D2']:
            pcts = {p: nbp.get(f'G24_{day}_P{p}') for p in [10, 50, 90]}
            print(f"  G24 {day} (mph): P10={pcts[10]}  P50={pcts[50]}  P90={pcts[90]}")
        print(f"  MaxT D1 P50={nbp.get('TMAX_D1_P50')}F P90={nbp.get('TMAX_D1_P90')}F  "
              f"D2 P50={nbp.get('TMAX_D2_P50')}F P90={nbp.get('TMAX_D2_P90')}F")
        print(f"  MinT N1 P50={nbp.get('TMIN_D1_P50')}F  N2 P50={nbp.get('TMIN_D2_P50')}F")

    blocks = make_blocks(nbh, nbs)
    block_hazards = [compute_block(b, i, nbp) for i, b in enumerate(blocks)]

    d1 = sum(1 for b in blocks[:8]  if b is not None)
    d2 = sum(1 for b in blocks[8:] if b is not None)
    print(f"  Timeline: {d1}/8 D1 blocks + {d2}/8 D2 blocks populated")

    # -- Threat cards: peak risk block over the full period -------------------
    threats = {}
    valid_indices = [i for i in range(16) if blocks[i]]

    for hz in HAZARDS:
        if not valid_indices:
            threats[hz] = {"prob":0,"risk":0,"risk_label":RISK_L[0],
                           "color":RISK_C[0],"level":0,"metric":"",
                           "peak_start_fxx":None,"peak_end_fxx":None,"peak_utc_start":None}
            continue

        risks  = [block_hazards[i][hz]["risk"]  for i in valid_indices]
        probs  = [block_hazards[i][hz]["prob"]  for i in valid_indices]
        levels = [block_hazards[i][hz]["level"] for i in valid_indices]

        # Peak = highest risk; break ties by highest probability
        pk = max(range(len(risks)), key=lambda i: (risks[i], probs[i]))
        mr  = risks[pk]
        mp  = probs[pk]
        plv = levels[pk]

        # TEMPERATURE: show only the applicable metric (heat or cold label)
        if hz == "TEMPERATURE":
            tt = block_hazards[valid_indices[pk]]["TEMPERATURE"].get("temp_type", "heat")
            met = METRICS["HEAT" if tt == "heat" else "COLD"].get(plv, "")
        else:
            met = METRICS.get(hz, {}).get(plv, "") if plv > 0 else ""

        pb = blocks[valid_indices[pk]]
        threats[hz] = {
            "prob": round(mp, 1), "risk": mr, "risk_label": RISK_L[mr],
            "color": RISK_C[mr], "level": plv, "metric": met,
            "peak_start_fxx": pb["start_fxx"] if pb else None,
            "peak_end_fxx":   pb["end_fxx"]   if pb else None,
            "peak_utc_start": pb["utc_start"]  if pb else None,
        }

    now_iso = now.isoformat()
    cycle_label = f"NBH {nbh_hs}Z / NBS {nbs_hs}Z / NBP {nbp_hs}Z"

    def serialize(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    # ISO timestamp of NBH cycle start — frontend uses this to compute correct peak times
    # peak_start_fxx is hours from cycle start, not from "now"
    cycle_utc_iso = (f"{nbh_ds[:4]}-{nbh_ds[4:6]}-{nbh_ds[6:8]}T{nbh_hs}:00:00Z"
                     if nbh_ds and nbh_hs else None)

    with open('threats.json', 'w') as f:
        json.dump({"threats": threats, "cycle": cycle_label,
                   "cycle_utc_iso": cycle_utc_iso, "last_updated": now_iso},
                  f, default=serialize)
    # Raw NBH hourly data for obs-panel forecast sparklines (1-hr resolution)
    nbh_hourly = []
    for fxx in range(1, 26):
        h = nbh.get(fxx, {})
        # WSP/GST/GSD from NBH are in KNOTS — convert to mph for frontend display
        def _kt(v): return round(v * KT_TO_MPH, 1) if v else 0
        nbh_hourly.append({
            'fxx': fxx, 'utc': h.get('utc_hour'),
            'TMP': h.get('TMP'), 'DPT': h.get('DPT'),
            'WDR': h.get('WDR'),
            'WSP': _kt(h.get('WSP')), 'GST': _kt(h.get('GST')), 'GSD': _kt(h.get('GSD')),
            'VIS': h.get('VIS'), 'SKY': h.get('SKY'),
            'P01': h.get('P01'), 'Q01': h.get('Q01'), 'T01': h.get('T01'),
            'MVV': h.get('MVV'), 'IFV': h.get('IFV'),
        })

    with open('timeline.json', 'w') as f:
        json.dump({"blocks": blocks, "block_hazards": block_hazards,
                   "nbh_hourly": nbh_hourly,
                   "cycle": cycle_label, "last_updated": now_iso},
                  f, default=serialize)


    # ── WIND card override: use NBP G24 (24-hr max gust distribution) ─────────
    # The block-level GST (instantaneous hourly gust) is correct for the timeline,
    # showing WHEN winds will be strongest. But for the card, G24 gives the proper
    # day-level aggregate: "what's the 24-hr maximum gust distribution?"
    # G24P5=48 mph correctly shows level-3 (45-58 mph) vs block GST=35 → level-2.
    for day_str, bi_range in [('D1', range(8)), ('D2', range(8, 16))]:
        p10 = nbp.get(f'G24_{day_str}_P10')
        p50 = nbp.get(f'G24_{day_str}_P50')
        p90 = nbp.get(f'G24_{day_str}_P90')
        if p50 is None: continue
        gm, gs = pct_to_gaussian(p10, p50, p90)
        if gm is None: continue
        if   gm >= 65: wl, low = 5, 65
        elif gm >= 58: wl, low = 4, 58
        elif gm >= 45: wl, low = 3, 45
        elif gm >= 30: wl, low = 2, 30
        else:          wl, low = 0, 30
        wp = gauss_above(gm, gs, low)
        if wl == 0 and wp < 5.0: wp, wl = 0.0, 0
        elif wl == 0: wl = 2
        if wl == 0: continue
        rk = risk_matrix(wp, wl)
        # Override if G24 gives equal or better result (higher level = more informative)
        curr = threats.get('WIND', {})
        if wl >= curr.get('level', 0):
            peak_bi = max(
                [i for i in bi_range if blocks[i] and blocks[i].get('GST') is not None],
                key=lambda i: blocks[i].get('GST', 0), default=None)
            pb = blocks[peak_bi] if peak_bi is not None else None
            threats['WIND'].update({
                "prob": round(wp, 1), "risk": rk, "risk_label": RISK_L[rk],
                "color": RISK_C[rk], "level": wl,
                "metric": METRICS.get("WIND", {}).get(wl, ""),
                # G24 percentiles already in mph (converted from kt in parse_nbp)
                "g24_p50_mph": round(gm, 1) if gm else None,
                "g24_p90_mph": round(p90, 1) if p90 else None,
            })
            if pb:
                threats['WIND'].update({
                    "peak_start_fxx": pb["start_fxx"],
                    "peak_end_fxx":   pb["end_fxx"],
                    "peak_utc_start": pb["utc_start"],
                })

    # ── TEMPERATURE card override: use NBP TXNP percentiles ──────────────────
    # Block TMP (hourly) correctly drives the timeline — shows when it's
    # actually hot/cold during specific periods. But for the card, the NBP
    # TXNP daily MaxT/MinT distribution captures the full probability range,
    # including the ~2°F gap between NBH hourly peak and the calibrated daily max.
    # Example: NBH peaks at 88°F (no heat risk per block), but NBP MaxT P50=90°F
    # → 50% chance of hitting the 90°F threshold → MINOR heat on the card.
    for day_str, bi_range in [('D1', range(8)), ('D2', range(8, 16))]:
        dk = '2' if day_str == 'D2' else '1'
        tmax_p50 = nbp.get(f'TMAX_D{dk}_P50')
        tmax_p10 = nbp.get(f'TMAX_D{dk}_P10')
        tmax_p90 = nbp.get(f'TMAX_D{dk}_P90')
        tmin_p50 = nbp.get(f'TMIN_D{dk}_P50')
        if tmax_p50 is None: continue

        # Heat
        maxt, maxt_std = pct_to_gaussian(tmax_p10, tmax_p50, tmax_p90)
        hp, hl = 0.0, 0
        if maxt:
            for t, l in [(90,2),(95,3),(100,4),(105,5)]:
                p = gauss_above(maxt, maxt_std or 3, t)
                if p >= 5.0: hp, hl = p, l
                else: break

        # Cold
        cp, cl = 0.0, 0
        if tmin_p50 is not None:
            mint_std = nbp.get(f'TMIN_D{dk}_STD') or 3
            for t, l in [(40,2),(32,3),(20,4),(0,5)]:
                p = gauss_below(tmin_p50, mint_std, t)
                if p >= 5.0: cp, cl = p, l
                else: break

        if risk_matrix(hp, hl) >= risk_matrix(cp, cl):
            card_p, card_l, met_key = hp, hl, 'HEAT'
        else:
            card_p, card_l, met_key = cp, cl, 'COLD'

        card_rk = risk_matrix(card_p, card_l)
        if card_l == 0: continue
        curr_temp = threats.get('TEMPERATURE', {})
        if card_rk >= curr_temp.get('risk', 0):
            # Timing: block with hottest (or coldest) TMP in this day range
            if met_key == 'HEAT':
                tmp_bi = max([i for i in bi_range if blocks[i] and blocks[i].get('TMP') is not None],
                             key=lambda i: blocks[i].get('TMP', -999), default=None)
            else:
                tmp_bi = min([i for i in bi_range if blocks[i] and blocks[i].get('TMP') is not None],
                             key=lambda i: blocks[i].get('TMP', 999), default=None)
            pb = blocks[tmp_bi] if tmp_bi is not None else None
            threats['TEMPERATURE'].update({
                "prob": round(card_p, 1), "risk": card_rk,
                "risk_label": RISK_L[card_rk], "color": RISK_C[card_rk],
                "level": card_l, "metric": METRICS.get(met_key, {}).get(card_l, ""),
            })
            if pb:
                threats['TEMPERATURE'].update({
                    "peak_start_fxx": pb["start_fxx"],
                    "peak_end_fxx":   pb["end_fxx"],
                    "peak_utc_start": pb["utc_start"],
                })

    print(f"\n  threats.json + timeline.json written  [{cycle_label}]")
    print("\nActive threats:")
    for hz, v in threats.items():
        if v["risk"] > 0:
            print(f"  {hz:15s}: {v['prob']:5.1f}%  {v['risk_label']:<14s}  {v['metric']}")

if __name__ == "__main__":
    main()
