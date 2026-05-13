#!/usr/bin/env python3
import re, json, requests, sys
from datetime import datetime, timezone, timedelta
from scipy.stats import norm
import numpy as np

KT_TO_MPH = 1.15078
STATION    = "KRNO"
NOMADS_NBM = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"

_MATRIX = {
    # Risk values from NOAA workflow matrix (image 2). Indexed by probability
    # bin (rows) and impact level 1-5 (cols, 0-indexed in the list).
    4: [1, 2, 3, 4, 5],   # Very Likely        (prob > 90%)
    3: [1, 2, 2, 3, 4],   # Likely             (66% < prob <= 90%)
    2: [1, 1, 2, 3, 4],   # As likely as not   (33% <= prob <= 66%)
    1: [1, 1, 2, 2, 3],   # Unlikely           (10% <= prob < 33%)
    0: [1, 1, 1, 2, 2],   # Extremely Unlikely (prob < 10%)
}

def risk_matrix(prob, level):
    """
    Compute risk rating (1-5) from probability % and impact level (1-5).
    Returns 0 if either input is 0 (i.e. hazard not in play at all).
    Probability bin boundaries match NOAA conventions:
      >90%   = Very Likely
      66-90% = Likely
      33-66% = As likely as not
      10-33% = Unlikely
      <10%   = Extremely Unlikely
    """
    if level == 0 or prob == 0: return 0
    if   prob >  90: pr = 4
    elif prob >  66: pr = 3
    elif prob >= 33: pr = 2
    elif prob >= 10: pr = 1
    else:            pr = 0
    return _MATRIX[pr][level - 1]

RISK_C = {0:"#3f3f46",1:"#e2f0cb",2:"#ffeb3b",3:"#ff9800",4:"#f44336",5:"#9c27b0"}
RISK_L = {0:"NONE",1:"LITTLE TO NONE",2:"MINOR",3:"MODERATE",4:"MAJOR",5:"EXTREME"}
HAZARDS = ["WIND","LIGHTNING","SNOW","VISIBILITY","FZRA","FLASH_FREEZE","RAIN","TEMPERATURE"]

METRICS = {
    "WIND":        {2:"30-45 mph",    3:"45-58 mph",    4:"58-65 mph",   5:">65 mph"},
    "SNOW":        {2:"T-0.5 in/hr",  3:"0.5-1 in/hr",  4:"1-2 in/hr",   5:">2 in/hr"},
    "LIGHTNING":   {1:"<5%",          2:"5-25%",        3:"25-50%",      4:"50-75%",     5:">75%"},
    "VISIBILITY":  {2:"3-5 SM",       3:"1-3 SM",       4:"0.5-1 SM",    5:"<0.5 SM"},
    "RAIN":        {2:"0.10-0.25 in/hr", 3:"0.25-0.50 in/hr", 4:"0.50-1.00 in/hr", 5:">1.00 in/hr"},
    "FZRA":        {2:"Trace",        3:"Trace-0.01 in", 4:"0.01-0.10 in", 5:">0.10 in"},
    "FLASH_FREEZE":{2:"Wet+Tw<36F",   3:"Wet+Tw<32F",   4:"Wet+Tw<28F",  5:"Wet+Tw<25F"},
    # HEAT and COLD share the TEMPERATURE display slot; the active set is
    # chosen by the temp_type field on the hazard record.
    "HEAT":        {2:"90-95F",       3:"95-100F",      4:"100-105F",    5:">105F"},
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
                'P01','Q01','T01','S01','I01',
                'PSN','PRA','PZR','PPL',
                'VIS','MVV','IFV','LIV',
                'CIG','MVC','IFC','LIC']
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
    g24p7 = parse_row('G24P7', sec, n_cols=n_cols)
    g24p9 = parse_row('G24P9', sec, n_cols=n_cols)
    txnp1 = parse_row('TXNP1', sec, n_cols=n_cols)
    txnp5 = parse_row('TXNP5', sec, n_cols=n_cols)
    txnp7 = parse_row('TXNP7', sec, n_cols=n_cols)
    txnp9 = parse_row('TXNP9', sec, n_cols=n_cols)
    
    # Temperature: store all percentiles for max and min
    if len(txnp5) >= 2 and txnp5[0] is not None:
        r.update({
            'TMAX_D1_P1': txnp1[0], 'TMAX_D1_P5': txnp5[0],
            'TMAX_D1_P7': txnp7[0], 'TMAX_D1_P9': txnp9[0],
            'TMIN_D1_P1': txnp1[1], 'TMIN_D1_P5': txnp5[1],
            'TMIN_D1_P7': txnp7[1], 'TMIN_D1_P9': txnp9[1],
        })
    
    # Wind gust: convert to mph and store all percentiles
    for pct, row in [(1, g24p1), (5, g24p5), (7, g24p7), (9, g24p9)]:
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
        def mx(k, default=0):
            v = [h[k] for h in nbh_hrs if h.get(k) is not None]
            return max(v) if v else default
        def mn(k, default=None):
            v = [h[k] for h in nbh_hrs if h.get(k) is not None]
            return min(v) if v else default
        blocks.append({
            'start_fxx': s, 'end_fxx': e, 'utc_start': nbh_hrs[0]['utc_hour'],
            # Temperature / moisture
            'TMP': av('TMP'), 'TSD': av('TSD'),
            'DPT': av('DPT'), 'DSD': av('DSD'),
            # Wind (knots -> mph)
            'WDR': nbh_hrs[0].get('WDR'),
            'WSP': round((av('WSP') or 0)*KT_TO_MPH, 1),
            'GST': round((av('GST') or 0)*KT_TO_MPH, 1),
            'GSD': round((av('GSD') or 3)*KT_TO_MPH, 1),
            # Sky / convection
            'SKY': av('SKY'),
            'T01': mx('T01'),              # peak thunderstorm prob in the 3h window
            # Precipitation: P01 peak, Q01 peak rate (was avg — wrong for hazard)
            'P01': mx('P01'),              # peak hourly precip prob
            'Q01': mx('Q01'),              # peak hourly QPF (1/100 in)
            'S01': mx('S01'),              # peak hourly snow (1/10 in)
            'I01': mx('I01'),              # peak hourly ice (1/100 in)
            # Conditional precip-type probabilities (peak across window)
            'PSN': mx('PSN'),
            'PRA': mx('PRA'),
            'PZR': mx('PZR'),
            'PPL': mx('PPL'),
            # Visibility (block-min in tenths of SM, NBM-direct probabilities)
            'VIS': mn('VIS', 100),
            'MVV': mx('MVV'),              # P(vis <= 5 SM)
            'IFV': mx('IFV'),              # P(vis <  3 SM)
            'LIV': mx('LIV'),              # P(vis <  1 SM)
            # Ceiling
            'CIG': mn('CIG', 999),
            'MVC': mx('MVC'),
            'IFC': mx('IFC'),
            'LIC': mx('LIC'),
        })
    return blocks

def _pack(prob, level, extra=None):
    """Build a single-hazard result dict with risk derived from prob+level."""
    prob = round(float(prob or 0), 1)
    risk = risk_matrix(prob, level)
    out = {"prob": prob, "risk": risk, "level": level, "color": RISK_C[risk]}
    if extra: out.update(extra)
    return out


def compute_block(block, bi, nbp, prev_block=None):
    """
    Per-block hazard computation.

    `prev_block` is the 3-hr block immediately before this one (or None for bi=0).
    It is used only by FLASH_FREEZE to detect "was wet, now freezing" sequences.
    """
    if not block:
        return {hz: _pack(0, 0) for hz in HAZARDS + ["COLD","HEAT"]}

    h = {}

    # ───── WIND ─────────────────────────────────────────────────────────────
    # Gaussian on peak gust (mean=GST, std=GSD). Pick the highest-risk threshold.
    gst, gsd = block.get('GST'), block.get('GSD')
    best_p, best_l, best_rk = 0.0, 0, 0
    for thr, lvl in [(65,5), (58,4), (45,3), (30,2)]:
        p  = gauss_above(gst, gsd, thr)
        rk = risk_matrix(p, lvl)
        if rk > best_rk or (rk == best_rk and p > best_p):
            best_p, best_l, best_rk = p, lvl, rk
    h["WIND"] = {"prob": best_p, "risk": best_rk, "level": best_l, "color": RISK_C[best_rk]}

    # ───── LIGHTNING ────────────────────────────────────────────────────────
    # Per NOAA table: L1=<5%, L2=5-25%, L3=25-50%, L4=50-75%, L5=>75%.
    # T01 is already P(thunder in the hour); we use the peak across the 3h window.
    t01 = float(block.get('T01') or 0)
    ll  = (5 if t01 >  75 else 4 if t01 >= 50 else 3 if t01 >= 25
           else 2 if t01 >= 5 else 1 if t01 > 0 else 0)
    h["LIGHTNING"] = _pack(t01, ll)

    # ───── PRECIP TYPE COMPOSITION ──────────────────────────────────────────
    # NBM gives unconditional P01 and conditional type fractions (PSN, PRA, PZR).
    # P(type) = P01 * P_type / 100.
    p01 = float(block.get('P01') or 0)
    psn = float(block.get('PSN') or 0)
    pra = float(block.get('PRA') or 0)
    pzr = float(block.get('PZR') or 0)
    p_snow = p01 * psn / 100.0
    p_rain = p01 * pra / 100.0
    p_fzra = p01 * pzr / 100.0

    # ───── SNOW ─────────────────────────────────────────────────────────────
    # Per NOAA table: L2=T-0.5"/hr, L3=0.5-1"/hr, L4=1-2"/hr, L5=>2"/hr.
    # Boundary values belong to the lower band (so 0.5 → L2, 1.0 → L3, 2.0 → L4).
    # S01 units = 1/10 inches per hour (so S01=5 → 0.5 in/hr).
    s01_tenths = float(block.get('S01') or 0)
    s_in_hr    = s01_tenths / 10.0
    snow_level = (5 if s_in_hr >  2.0
                  else 4 if s_in_hr >  1.0
                  else 3 if s_in_hr >  0.5
                  else 2 if s_in_hr >  0     # trace - 0.5 in/hr
                  else 0)
    # Even if no measurable S01, if snow precip-type chance is nontrivial it's
    # still possibly snowing — register at L2 (trace) so the hazard is on radar.
    if snow_level == 0 and p_snow >= 10:
        snow_level = 2
    h["SNOW"] = _pack(p_snow, snow_level, {"rate_in_hr": round(s_in_hr, 2)})

    # ───── RAIN ─────────────────────────────────────────────────────────────
    # Per NOAA table: L2=0.10-0.25"/hr, L3=0.25-0.50"/hr, L4=0.50-1.00"/hr, L5=>1.00"/hr.
    # Lower-bound values belong to the higher band (so 0.10→L2, 0.25→L3, 0.50→L4, 1.00→L5).
    # Q01 units = 1/100 inches per hour (so Q01=10 → 0.10"/hr).
    q01_hundredths = float(block.get('Q01') or 0)
    r_in_hr        = q01_hundredths / 100.0
    rain_level = (5 if r_in_hr >= 1.00
                  else 4 if r_in_hr >= 0.50
                  else 3 if r_in_hr >= 0.25
                  else 2 if r_in_hr >= 0.10
                  else 0)
    # Type-prob fallback: if rain forecast is meaningful but Q01 didn't reach
    # the L2 floor, still register at L2 so the hazard appears on radar.
    if rain_level == 0 and p_rain >= 20 and r_in_hr > 0:
        rain_level = 2
    h["RAIN"] = _pack(p_rain, rain_level, {"rate_in_hr": round(r_in_hr, 3)})

    # ───── FREEZING RAIN ────────────────────────────────────────────────────
    # Per NOAA table: L2=Trace (expected), L3=Trace-0.01", L4=0.01-0.10", L5=>0.10".
    # I01 units = 1/100 inches ice per hour.
    i01_hundredths = float(block.get('I01') or 0)
    ice_in_hr      = i01_hundredths / 100.0
    fzra_level = (5 if ice_in_hr >  0.10
                  else 4 if ice_in_hr >  0.01
                  else 3 if ice_in_hr >  0
                  else 2 if p_fzra >= 5 else 0)   # Trace expected
    h["FZRA"] = _pack(p_fzra, fzra_level, {"rate_in_hr": round(ice_in_hr, 3)})

    # ───── VISIBILITY ───────────────────────────────────────────────────────
    # Per NOAA table: L2=3-5 SM, L3=1-3 SM, L4=0.50-1 SM, L5=<0.50 SM.
    # NBM provides exceedance probabilities directly:
    #   MVV = P(vis <= 5 SM)  — corresponds to L2 (or worse)
    #   IFV = P(vis <  3 SM)  — corresponds to L3 (or worse)
    #   LIV = P(vis <  1 SM)  — corresponds to L4 (or worse)
    # NBM does not provide a P(vis < 0.5 SM) field; approximate using the
    # block-min VIS as a constraint (LIV is a superset of the <0.5 event).
    mvv = float(block.get('MVV') or 0)
    ifv = float(block.get('IFV') or 0)
    liv = float(block.get('LIV') or 0)
    vis_t = block.get('VIS')   # tenths of SM (5 = 0.5 SM, 10 = 1.0 SM)
    if vis_t is not None and vis_t < 5:
        # Forecast already indicates < 0.5 SM — use LIV as a lower bound for VLIFR
        vlifr_p = liv
    elif vis_t is not None and vis_t < 10:
        # Forecast in the 0.5–1.0 SM range — soften LIV downward
        vlifr_p = liv * 0.5
    else:
        vlifr_p = 0
    # Pick the highest-risk band across all four
    best_vp, best_vl, best_vr = 0, 0, 0
    for prob, lvl in [(vlifr_p, 5), (liv, 4), (ifv, 3), (mvv, 2)]:
        rk = risk_matrix(prob, lvl)
        if rk > best_vr or (rk == best_vr and prob > best_vp):
            best_vp, best_vl, best_vr = prob, lvl, rk
    h["VISIBILITY"] = {"prob": round(best_vp, 1), "risk": best_vr,
                       "level": best_vl, "color": RISK_C[best_vr]}

    # ───── FLASH FREEZE ─────────────────────────────────────────────────────
    # Per NOAA table: L2=Wet+<36°F Tw, L3=Wet+<32°F Tw, L4=Wet+<28°F Tw, L5=Wet+<25°F Tw.
    # (L1 = "Dry and >32°F Tw" = no flash-freeze hazard.)
    # We treat "wet" as P(precip occurring or recently occurred) >= 30%.
    # Joint probability: P(wet) × P(Tw <= threshold).
    tmp, tsd = block.get('TMP'), (block.get('TSD') or 3)
    dpt      = block.get('DPT')
    tw       = None
    if tmp is not None and dpt is not None:
        # 1/3 rule approximation: Tw ≈ Tmp - (Tmp - Dpt)/3 (°F)
        tw = tmp - (tmp - dpt) / 3.0
    p01_prev = float((prev_block or {}).get('P01') or 0) if prev_block else 0
    p_wet    = max(p01, p01_prev)

    ff_level, ff_prob = 0, 0
    if tw is not None and p_wet >= 30:
        # Use Tw mean with TSD as proxy for Tw std (DSD is similar magnitude
        # and Tw is dominated by TMP for typical T-Td spreads).
        for thr, lvl in [(25, 5), (28, 4), (32, 3), (36, 2)]:
            p_freeze = gauss_below(tw, tsd, thr)
            composite = (p_wet / 100.0) * p_freeze    # both inputs in %, composite stays in %
            rk = risk_matrix(composite, lvl)
            if rk > risk_matrix(ff_prob, ff_level):
                ff_prob, ff_level = composite, lvl
    h["FLASH_FREEZE"] = _pack(ff_prob, ff_level,
                              {"wet_pct": round(p_wet, 1),
                               "tw_F": round(tw, 1) if tw is not None else None})

    # ───── TEMPERATURE (heat OR cold) ───────────────────────────────────────
    # Per NOAA table:
    #   COLD: L2=32-40°F, L3=20-32°F, L4=10-20°F, L5=<10°F.  (L1 = ≥40°F.)
    #   HEAT: L2=90-95°F, L3=95-100°F, L4=100-105°F, L5=>105°F.  (L1 = <90°F.)
    # Walk thresholds and pick the level that yields the highest risk (matches
    # how WIND handles its bands), not just "last threshold that fires".
    tmp, tsd = block.get('TMP'), (block.get('TSD') or 3)
    cp, cl, c_rk = 0, 0, 0
    for thr, lvl in [(40, 2), (32, 3), (20, 4), (10, 5)]:
        p = gauss_below(tmp, tsd, thr)
        rk = risk_matrix(p, lvl)
        if rk > c_rk or (rk == c_rk and p > cp):
            cp, cl, c_rk = p, lvl, rk
    hp, hl, h_rk = 0, 0, 0
    for thr, lvl in [(90, 2), (95, 3), (100, 4), (105, 5)]:
        p = gauss_above(tmp, tsd, thr)
        rk = risk_matrix(p, lvl)
        if rk > h_rk or (rk == h_rk and p > hp):
            hp, hl, h_rk = p, lvl, rk
    if h_rk >= c_rk:
        h["TEMPERATURE"] = _pack(hp, hl, {"temp_type": "heat"})
    else:
        h["TEMPERATURE"] = _pack(cp, cl, {"temp_type": "cold"})

    return h

def main():
    ds, hs, url = get_cycle('h')
    nbh_sec = fetch_station(url)
    nbp_url = get_cycle('p')[2]
    nbp_sec = fetch_station(nbp_url) if nbp_url else None
    if not nbh_sec: sys.exit(1)
    nbh, nbp = parse_nbh(nbh_sec), parse_nbp(nbp_sec) if nbp_sec else {}
    # DEBUG
    print(f"DEBUG: NBP URL: {nbp_url}", file=sys.stderr)
    print(f"DEBUG: NBP section fetched: {nbp_sec is not None}", file=sys.stderr)
    print(f"DEBUG: NBP dict keys: {list(nbp.keys())}", file=sys.stderr)
    if 'G24_D1_P5' in nbp:
        print(f"DEBUG: G24_D1_P5 = {nbp['G24_D1_P5']}", file=sys.stderr)
    blocks = make_blocks(nbh, {})
    block_hazards = [compute_block(b, i, nbp, prev_block=blocks[i-1] if i > 0 else None)
                     for i, b in enumerate(blocks)]
    threats = {}
    for hz in HAZARDS:
        idx = [i for i in range(len(blocks)) if blocks[i]]
        pk = max(idx, key=lambda i: (block_hazards[i][hz]["risk"], block_hazards[i][hz]["prob"]))
        pk_hz = block_hazards[pk][hz]
        # For TEMPERATURE, route to HEAT or COLD metric labels via the
        # temp_type discriminator the compute_block sets.
        if hz == "TEMPERATURE":
            metric_key = "HEAT" if pk_hz.get("temp_type") == "heat" else "COLD"
        else:
            metric_key = hz
        
        # Custom display thresholds per hazard
        display = False
        if hz == "LIGHTNING":
            # Show lightning if 0% < prob < 5% (low but operationally important)
            display = 0 < pk_hz["prob"] < 5
        elif hz == "WIND":
            # Show wind only if risk >= 2 (MINOR or higher, i.e., >=30 mph threat)
            display = pk_hz["risk"] >= 2
        elif hz == "RAIN":
            # Show rain if it's between 0.01" and 0.10" (trace to light)
            # Rain level corresponds to rate_in_hr in hundredths of inches
            rain_rate = pk_hz.get("rate_in_hr", 0)
            display = 0.01 <= rain_rate <= 0.10
        else:
            # All other hazards: show if risk >= 2 (MINOR or higher)
            display = pk_hz["risk"] >= 2
        
        if display:
            threats[hz] = {
                "prob": pk_hz["prob"], "risk": pk_hz["risk"],
                "risk_label": RISK_L[pk_hz["risk"]], "color": pk_hz["color"],
                "level": pk_hz["level"],
                "metric": METRICS.get(metric_key, {}).get(pk_hz["level"], ""),
                "peak_start_fxx": blocks[pk]["start_fxx"],
                "peak_end_fxx": blocks[pk]["end_fxx"],
                "peak_utc_start": blocks[pk]["utc_start"],
            }
            # Surface temp_type so the frontend can label "Heat" vs "Cold" if it wants
            if hz == "TEMPERATURE" and "temp_type" in pk_hz:
                threats[hz]["temp_type"] = pk_hz["temp_type"]
        else:
            # Not displayed: create minimal entry with risk=0 so frontend knows it exists but isn't active
            threats[hz] = {
                "prob": 0.0, "risk": 0,
                "risk_label": "NONE", "color": "#3f3f46",
                "level": 0, "metric": "",
                "peak_start_fxx": blocks[pk]["start_fxx"],
                "peak_end_fxx": blocks[pk]["end_fxx"],
                "peak_utc_start": blocks[pk]["utc_start"],
            }

    # ───── WIND OVERRIDE WITH 24H GUST PERCENTILE ──────────────────────────
    # Use NBP's G24P5 (24-hour gust percentile) instead of 3-hour block average,
    # since it's more accurate and matches operational briefing data. The peak
    # timing still comes from the hourly block (when the peak gust occurs).
    g24p5 = nbp.get('G24_D1_P5')  # 24h median gust in mph
    if g24p5 is not None:
        g24_std = 3  # assume modest std for percentile forecast
        best_p, best_l, best_rk = 0.0, 0, 0
        for thr, lvl in [(65, 5), (58, 4), (45, 3), (30, 2)]:
            p = gauss_above(g24p5, g24_std, thr)
            rk = risk_matrix(p, lvl)
            if rk > best_rk or (rk == best_rk and p > best_p):
                best_p, best_l, best_rk = p, lvl, rk
        if best_rk >= 2:  # Only override if there's MINOR or higher wind risk
            threats['WIND'].update({
                "prob": best_p, "risk": best_rk, "risk_label": RISK_L[best_rk],
                "color": RISK_C[best_rk], "level": best_l,
                "metric": METRICS["WIND"].get(best_l, ""),
                "g24_d1_p50_mph": g24p5  # For reference in frontend
            })
            # Find the block containing the actual peak gust hour (highest GST in NBH)
            peak_gust_hour = 1
            peak_gust_val = 0
            for fxx, hdata in nbh.items():
                if hdata and hdata.get('GST') and hdata['GST'] > peak_gust_val:
                    peak_gust_val = hdata['GST']
                    peak_gust_hour = fxx
            # Determine which block this hour belongs to (block bi spans fxx 3*bi+1 to 3*bi+3)
            pk_block_idx = (peak_gust_hour - 1) // 3
            if 0 <= pk_block_idx < len(block_hazards) and block_hazards[pk_block_idx]:
                block_hazards[pk_block_idx]['WIND'] = threats['WIND'].copy()

    # ───── TEMPERATURE OVERRIDE WITH 24H MAX/MIN ─────────────────────────
    # Use NBP's TXN (24-hour max/min) instead of block averages for the
    # threat matrix, since they're more accurate. The peak timing still comes
    # from the hourly block (when it actually occurs during the day).
    # Only show if risk >= 2 (MINOR or higher), not LITTLE-TO-NONE (risk=1).
    tmax_d1 = nbp.get('TMAX_D1_P5')
    tmin_d1 = nbp.get('TMIN_D1_P5')
    if tmax_d1 is not None or tmin_d1 is not None:
        # Compute heat hazard from tmax_d1
        if tmax_d1 is not None:
            tmax_std = 3  # assume small std for point forecast
            hp, hl, h_rk = 0, 0, 0
            for thr, lvl in [(90, 2), (95, 3), (100, 4), (105, 5)]:
                p = gauss_above(tmax_d1, tmax_std, thr)
                rk = risk_matrix(p, lvl)
                if rk > h_rk or (rk == h_rk and p > hp):
                    hp, hl, h_rk = p, lvl, rk
            if h_rk >= 2:  # Only override if there's MINOR or higher heat risk
                threats['TEMPERATURE'].update({
                    "prob": hp, "risk": h_rk, "risk_label": RISK_L[h_rk],
                    "color": RISK_C[h_rk], "level": hl,
                    "metric": METRICS["HEAT"].get(hl, ""),
                    "temp_type": "heat",
                    "txn_24h_max": tmax_d1
                })
                # Find the block containing the actual peak MAX temp hour
                peak_max_hour = 1
                peak_max_val = -999
                for fxx, hdata in nbh.items():
                    if hdata and hdata.get('TMP') and hdata['TMP'] > peak_max_val:
                        peak_max_val = hdata['TMP']
                        peak_max_hour = fxx
                pk_max_block_idx = (peak_max_hour - 1) // 3
                if 0 <= pk_max_block_idx < len(block_hazards) and block_hazards[pk_max_block_idx]:
                    block_hazards[pk_max_block_idx]['TEMPERATURE'] = threats['TEMPERATURE'].copy()
        # Compute cold hazard from tmin_d1
        if tmin_d1 is not None:
            tmin_std = 3
            cp, cl, c_rk = 0, 0, 0
            for thr, lvl in [(40, 2), (32, 3), (20, 4), (10, 5)]:
                p = gauss_below(tmin_d1, tmin_std, thr)
                rk = risk_matrix(p, lvl)
                if rk > c_rk or (rk == c_rk and p > cp):
                    cp, cl, c_rk = p, lvl, rk
            if c_rk >= 2:  # Only override if there's MINOR or higher cold risk
                threats['TEMPERATURE'].update({
                    "prob": cp, "risk": c_rk, "risk_label": RISK_L[c_rk],
                    "color": RISK_C[c_rk], "level": cl,
                    "metric": METRICS["COLD"].get(cl, ""),
                    "temp_type": "cold",
                    "txn_24h_min": tmin_d1
                })
                # Find the block containing the actual peak MIN temp hour
                peak_min_hour = 1
                peak_min_val = 999
                for fxx, hdata in nbh.items():
                    if hdata and hdata.get('TMP') and hdata['TMP'] < peak_min_val:
                        peak_min_val = hdata['TMP']
                        peak_min_hour = fxx
                pk_min_block_idx = (peak_min_hour - 1) // 3
                if 0 <= pk_min_block_idx < len(block_hazards) and block_hazards[pk_min_block_idx]:
                    block_hazards[pk_min_block_idx]['TEMPERATURE'] = threats['TEMPERATURE'].copy()



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
