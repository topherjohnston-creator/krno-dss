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

RISK_C = {0:"#27272a",1:"#3f3f46",2:"#ffeb3b",3:"#ff9800",4:"#f44336",5:"#9c27b0"}
RISK_L = {0:"NONE",1:"LITTLE TO NONE",2:"MINOR",3:"MODERATE",4:"MAJOR",5:"EXTREME"}
HAZARDS = ["WIND","LIGHTNING","SNOW","VISIBILITY","FZRA","FLASH_FREEZE","RAIN","TEMPERATURE"]

METRICS = {
    "WIND":        {2:"30-45 mph",    3:"45-58 mph",    4:"58-65 mph",   5:">65 mph"},
    "SNOW":        {2:"Trace-2in/24hr", 3:"2-4in/12hr or 3-6in/24hr", 4:"4+in/12hr or 6+in/24hr", 5:"Well above warning"},
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
    if not sec: return {}
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
    s24p1 = parse_row('S24P1', sec, n_cols=n_cols)
    s24p5 = parse_row('S24P5', sec, n_cols=n_cols)
    s24p9 = parse_row('S24P9', sec, n_cols=n_cols)

    # ── Day 1 (col 0 = max, col 1 = min) ──
    if txnp5 and len(txnp5) > 0 and txnp5[0] is not None:
        r.update({'TMAX_D1_P1': txnp1[0], 'TMAX_D1_P5': txnp5[0],
                  'TMAX_D1_P7': txnp7[0], 'TMAX_D1_P9': txnp9[0]})
    if txnp5 and len(txnp5) > 1 and txnp5[1] is not None:
        r.update({'TMIN_D1_P1': txnp1[1], 'TMIN_D1_P5': txnp5[1],
                  'TMIN_D1_P7': txnp7[1], 'TMIN_D1_P9': txnp9[1]})
    for pct, row in [(1,g24p1),(5,g24p5),(7,g24p7),(9,g24p9)]:
        if row and len(row) > 0 and row[0] is not None:
            r[f'G24_D1_P{pct}'] = round(row[0] * KT_TO_MPH, 1)
    for pct, row in [(1,s24p1),(5,s24p5),(9,s24p9)]:
        if row and len(row) > 0 and row[0] is not None:
            r[f'S24_D1_P{pct}'] = round(row[0] / 10.0, 2)

    # ── Day 2 (col 2 = max, col 3 = min; G24/S24 col 1) ──
    if txnp5 and len(txnp5) > 2 and txnp5[2] is not None:
        r.update({'TMAX_D2_P1': txnp1[2], 'TMAX_D2_P5': txnp5[2],
                  'TMAX_D2_P7': txnp7[2], 'TMAX_D2_P9': txnp9[2]})
    if txnp5 and len(txnp5) > 3 and txnp5[3] is not None:
        r.update({'TMIN_D2_P1': txnp1[3], 'TMIN_D2_P5': txnp5[3],
                  'TMIN_D2_P7': txnp7[3], 'TMIN_D2_P9': txnp9[3]})
    for pct, row in [(1,g24p1),(5,g24p5),(7,g24p7),(9,g24p9)]:
        if row and len(row) > 1 and row[1] is not None:
            r[f'G24_D2_P{pct}'] = round(row[1] * KT_TO_MPH, 1)
    for pct, row in [(1,s24p1),(5,s24p5),(9,s24p9)]:
        if row and len(row) > 1 and row[1] is not None:
            r[f'S24_D2_P{pct}'] = round(row[1] / 10.0, 2)

    return r


def parse_nbs(sec):
    """Parse NBS (3-hourly, hours 5-71) bulletin into a dict keyed by FHR (5,8,11,...)."""
    if not sec: return {}
    n_cols, utc_row = _utc_col_count(sec)
    elements = ['TMP','TSD','DPT','DSD','WDR','WSP','GST','GSD','SKY',
                'P06','Q06','T03','S06','I06',
                'PSN','PRA','PZR','PPL',
                'VIS','IFV','MVV']
    rows = {el: parse_row(el, sec, n_cols=n_cols) for el in elements}
    data = {}
    for i in range(n_cols):
        fxx_val = 5 + i * 3  # FHR: 5,8,11,...71
        entry = {'utc_hour': utc_row[i] if i < len(utc_row) else None, 'fxx': fxx_val}
        for el, vals in rows.items():
            entry[el] = vals[i] if i < len(vals) else None
        data[fxx_val] = entry
    return data

def make_blocks(nbh, nbs):
    """Build 16 × 3-hr blocks. Blocks 0-7 from NBH (fxx 1-24), blocks 8-15 from NBS (fxx 26-48)."""
    if not nbh.get(1): return [None]*16
    blocks = []
    for bi in range(16):
        s, e = bi * 3 + 1, bi * 3 + 3

        if bi < 8:
            # Day 1: use NBH hourly data
            nbh_hrs = [nbh.get(f) for f in range(s, e + 1) if nbh.get(f)]
            if not nbh_hrs: blocks.append(None); continue
            src = nbh_hrs
            def av(k):
                v = [h[k] for h in src if h.get(k) is not None]
                return sum(v)/len(v) if v else None
            def mx(k, default=0):
                v = [h[k] for h in src if h.get(k) is not None]
                return max(v) if v else default
            def mn(k, default=None):
                v = [h[k] for h in src if h.get(k) is not None]
                return min(v) if v else default
            # Use T03 from NBS for this block using nearest FHR (5,8,11,...23)
            mid = (s + e) / 2
            nearest_fhr = min([5,8,11,14,17,20,23], key=lambda f: abs(f - mid))
            nbs_match = nbs.get(nearest_fhr) if nbs else None
            t03_val   = nbs_match.get('T03') if nbs_match else None
            blocks.append({
                'start_fxx': s, 'end_fxx': e, 'utc_start': src[0]['utc_hour'],
                'TMP': av('TMP'), 'TSD': av('TSD'),
                'DPT': av('DPT'), 'DSD': av('DSD'),
                'WDR': src[0].get('WDR'),
                'WSP': round((av('WSP') or 0)*KT_TO_MPH, 1),
                'GST': round((av('GST') or 0)*KT_TO_MPH, 1),
                'GSD': round((av('GSD') or 3)*KT_TO_MPH, 1),
                'SKY': av('SKY'),
                'T01': t03_val if t03_val is not None else mx('T01'),
                'P01': mx('P01'),
                'Q01': mx('Q01'),
                'S01': sum((h.get('S01') or 0) for h in src),
                's3hr_in': round(sum((h.get('S01') or 0) for h in src) / 10.0, 2),
                'I01': mx('I01'),
                'PSN': mx('PSN'), 'PRA': mx('PRA'), 'PZR': mx('PZR'), 'PPL': mx('PPL'),
                'VIS': mn('VIS', 100),
                'MVV': mx('MVV'), 'IFV': mx('IFV'), 'LIV': mx('LIV'),
                'CIG': mn('CIG', 999), 'MVC': mx('MVC'), 'IFC': mx('IFC'),
                'source': 'NBH'
            })
        else:
            # Day 2: use NBS 3-hourly data
            # NBS fxx for this block: blocks 8-15 map to fxx 26,29,32,...47
            nbs_fxx = 26 + (bi - 8) * 3
            nb = nbs.get(nbs_fxx) if nbs else None
            if not nb: blocks.append(None); continue
            def _kt(v): return round((v or 0)*KT_TO_MPH, 1)
            blocks.append({
                'start_fxx': nbs_fxx, 'end_fxx': nbs_fxx + 2,
                'utc_start': nb.get('utc_hour'),
                'TMP': nb.get('TMP'), 'TSD': nb.get('TSD'),
                'DPT': nb.get('DPT'), 'DSD': nb.get('DSD'),
                'WDR': nb.get('WDR'),
                'WSP': _kt(nb.get('WSP')),
                'GST': _kt(nb.get('GST')),
                'GSD': _kt(nb.get('GSD')),
                'SKY': nb.get('SKY'),
                'T01': nb.get('T03'),          # NBS uses T03 (3hr lightning prob)
                'P01': nb.get('P06'),          # NBS uses P06
                'Q01': nb.get('Q06'),          # NBS uses Q06
                'S01': round((nb.get('S06') or 0) / 2, 2),  # S06 split across 2 blocks
                's3hr_in': round((nb.get('S06') or 0) / 2 / 10.0, 2),  # 3hr snow in inches
                'I01': nb.get('I06'),
                'PSN': nb.get('PSN'), 'PRA': nb.get('PRA'),
                'PZR': nb.get('PZR'), 'PPL': nb.get('PPL'),
                'VIS': nb.get('VIS'),
                'MVV': nb.get('MVV'), 'IFV': nb.get('IFV'), 'LIV': None,
                'CIG': None, 'MVC': None, 'IFC': None,
                'source': 'NBS'
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
    Per-block hazard computation using NBH block mean values.
    Main() applies NBP percentile overrides to the threat matrix and
    peak blocks after this runs.
    `prev_block` used only by FLASH_FREEZE for wet-then-freeze detection.
    """
    if not block:
        return {hz: _pack(0, 0) for hz in HAZARDS + ["COLD","HEAT"]}

    h = {}

    # ───── WIND ─────────────────────────────────────────────────────────────
    # Gaussian on block mean gust. Main() will override with P50 percentile
    # for the threat matrix and stamped blocks after compute_block runs.
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
    p_rain = p01 * pra / 100.0
    p_fzra = p01 * pzr / 100.0

    # ───── SNOW ─────────────────────────────────────────────────────────────
    # S01 is now the SUM of hourly snow across the 3hr block (1/10ths of an inch).
    # Convert directly to inches — no multiplication needed.
    # Level thresholds based on advisory/warning criteria:
    #   L2 (MINOR):   Trace — any measurable snow
    #   L3 (MODERATE): Advisory pace — ~0.5"+ per 3hr (≈2-4"/12hr)
    #   L4 (MAJOR):   Warning pace  — ~1.0"+ per 3hr (≈4"/12hr)
    #   L5 (EXTREME): Well above warning — ~2.0"+ per 3hr
    s01_tenths = float(block.get('S01') or 0)
    s3hr_in    = s01_tenths / 10.0
    snow_level = (5 if s3hr_in >= 2.0
                  else 4 if s3hr_in >= 1.0
                  else 3 if s3hr_in >= 0.5
                  else 2 if s3hr_in >  0.0
                  else 0)
    # If no accumulation but PSN is meaningful, flag as trace (L2)
    if snow_level == 0 and psn >= 10:
        snow_level = 2
    # No probability for snow blocks — level IS the forecast. Use 100% so
    # risk matrix gives full weight to the level and color is driven by severity.
    snow_prob = 100.0 if snow_level > 0 else 0.0
    h["SNOW"] = _pack(snow_prob, snow_level, {"s3hr_in": round(s3hr_in, 2)})

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
    # ───── TEMPERATURE (heat OR cold) ───────────────────────────────────────
    # Use block mean TMP with Gaussian. Main() will override with NBP P50
    # percentile for the threat matrix and stamped blocks after compute_block runs.
    # Sub-1% probabilities floored to 0 to avoid spurious LITTLE-TO-NONE.
    tmp, tsd = block.get('TMP'), (block.get('TSD') or 3)
    cp, cl, c_rk = 0, 0, 0
    for thr, lvl in [(40,2),(32,3),(20,4),(10,5)]:
        p = gauss_below(tmp, tsd, thr)
        if p < 1.0: p = 0.0
        rk = risk_matrix(p, lvl)
        if rk > c_rk or (rk == c_rk and p > cp):
            cp, cl, c_rk = p, lvl, rk
    hp, hl, h_rk = 0, 0, 0
    for thr, lvl in [(90,2),(95,3),(100,4),(105,5)]:
        p = gauss_above(tmp, tsd, thr)
        if p < 1.0: p = 0.0
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
    nbs_cycle = get_cycle('s')
    nbs_url   = nbs_cycle[2] if nbs_cycle[0] else None
    nbs_sec   = fetch_station(nbs_url) if nbs_url else None
    if not nbh_sec: sys.exit(1)
    nbh  = parse_nbh(nbh_sec)
    nbp  = parse_nbp(nbp_sec) if nbp_sec else {}
    nbs  = parse_nbs(nbs_sec) if nbs_sec else {}
    blocks = make_blocks(nbh, nbs)
    
    # Find peak gust hour and peak max/min temp hours BEFORE computing blocks
    peak_gust_hour = 1
    peak_gust_val = 0
    peak_max_hour = 1
    peak_max_val = -999
    peak_min_hour = 1
    peak_min_val = 999
    for fxx, hdata in nbh.items():
        if hdata:
            gst = hdata.get('GST')
            tmp = hdata.get('TMP')
            if gst and gst > peak_gust_val:
                peak_gust_val = gst
                peak_gust_hour = fxx
            if tmp:
                if tmp > peak_max_val:
                    peak_max_val = tmp
                    peak_max_hour = fxx
                if tmp < peak_min_val:
                    peak_min_val = tmp
                    peak_min_hour = fxx
    
    # Convert hours to block indices
    peak_gust_block_idx = (peak_gust_hour - 1) // 3
    peak_max_block_idx  = (peak_max_hour  - 1) // 3
    peak_min_block_idx  = (peak_min_hour  - 1) // 3
    
    # Compute block hazards with peak flags
    block_hazards = []
    for i, b in enumerate(blocks):
        block_hazards.append(compute_block(b, i, nbp, prev_block=blocks[i-1] if i > 0 else None))
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
        
        # Always write the actual computed prob/risk — frontend dims the card if risk=0 or risk=1
        threats[hz] = {
            "prob": pk_hz["prob"], "risk": pk_hz["risk"],
            "risk_label": RISK_L[pk_hz["risk"]], "color": RISK_C[pk_hz["risk"]],
            "level": pk_hz["level"],
            "metric": METRICS.get(metric_key, {}).get(pk_hz["level"], ""),
            "peak_start_fxx": blocks[pk]["start_fxx"],
            "peak_end_fxx": blocks[pk]["end_fxx"],
            "peak_utc_start": blocks[pk]["utc_start"],
        }
        if hz == "TEMPERATURE" and "temp_type" in pk_hz:
            threats[hz]["temp_type"] = pk_hz["temp_type"]

    # ───── WIND OVERRIDE WITH 24H GUST PERCENTILE ──────────────────────────
    # Use G24P9 (90th pct = actual peak) for prob computation in threat matrix.
    # Also stamp the peak gust block in the timeline with the same value.
    g24p1  = nbp.get('G24_D1_P1')
    g24p5  = nbp.get('G24_D1_P5')
    g24p9  = nbp.get('G24_D1_P9')
    if g24p5 is not None:
        # NOAA workflow: compute P(gust > threshold) for each impact level,
        # look up risk matrix, pick highest risk.
        # Use percentile interpolation: P1=10th, P5=50th, P9=90th percentile.
        pts = sorted([(v,c) for v,c in [(g24p1,10),(g24p5,50),(g24p9,90)] if v is not None])

        def pct_above(thr):
            if not pts: return 0.0
            if thr <= pts[0][0]: return round(100.0 - pts[0][1], 1)
            if thr > pts[-1][0]: return 0.0  # threshold exceeds P90 — essentially 0% chance
            for i in range(len(pts)-1):
                v0,c0 = pts[i]; v1,c1 = pts[i+1]
                if v0 <= thr <= v1:
                    pct = c0 + (thr-v0)*(c1-c0)/(v1-v0)
                    return round(max(0.0, 100.0 - pct), 1)
            return 0.0

        best_p, best_l, best_rk = 0.0, 0, 0
        for thr, lvl in [(65,5),(58,4),(45,3),(30,2)]:
            p  = pct_above(thr)
            rk = risk_matrix(p, lvl)
            if rk > best_rk or (rk == best_rk and p > best_p):
                best_p, best_l, best_rk = p, lvl, rk

        # Always write percentiles so wind rose tooltip can show distribution
        threats['WIND'].update({
            "g24_p10_mph": g24p1,
            "g24_p50_mph": g24p5,
            "g24_p90_mph": g24p9,
        })

        # Always update prob/risk from 24h computation so card shows correct probability
        pk_blk = blocks[peak_gust_block_idx] if peak_gust_block_idx < len(blocks) and blocks[peak_gust_block_idx] else None
        threats['WIND'].update({
            "prob": best_p, "risk": best_rk, "risk_label": RISK_L[best_rk],
            "color": RISK_C[best_rk], "level": best_l,
            "metric": METRICS["WIND"].get(best_l, ""),
            **({"peak_start_fxx": pk_blk["start_fxx"],
                "peak_end_fxx":   pk_blk["end_fxx"],
                "peak_utc_start": pk_blk["utc_start"]} if pk_blk else {})
        })

        if best_rk >= 2:
            pk_blk = blocks[peak_gust_block_idx] if peak_gust_block_idx < len(blocks) and blocks[peak_gust_block_idx] else None
            threats['WIND'].update({
                "prob": best_p, "risk": best_rk, "risk_label": RISK_L[best_rk],
                "color": RISK_C[best_rk], "level": best_l,
                "metric": METRICS["WIND"].get(best_l, ""),
                **({"peak_start_fxx": pk_blk["start_fxx"],
                    "peak_end_fxx":   pk_blk["end_fxx"],
                    "peak_utc_start": pk_blk["utc_start"]} if pk_blk else {})
            })
            # Stamp all blocks in the wind event window.
            gst_floor = g24p5 * 0.5
            stamped_blocks = set()
            for bi, blk in enumerate(blocks):
                if blk and blk.get('GST', 0) >= gst_floor:
                    stamped_blocks.add(bi)
            stamped_blocks.add(peak_gust_block_idx)
            for bi in stamped_blocks:
                if 0 <= bi < len(block_hazards) and block_hazards[bi]:
                    block_hazards[bi]['WIND'] = {
                        "prob": best_p, "risk": best_rk, "level": best_l, "color": RISK_C[best_rk]
                    }

    # ───── SNOW OVERRIDE WITH 24H ACCUMULATION PERCENTILE ───────────────────
    # Use S24P1/P5/P9 from NBP for threat matrix card.
    # Same NOAA workflow: P(snow > threshold) for each level, risk matrix.
    # Thresholds based on advisory/warning criteria (24hr accumulation in inches).
    s24p1 = nbp.get('S24_D1_P1')
    s24p5 = nbp.get('S24_D1_P5')
    s24p9 = nbp.get('S24_D1_P9')
    if s24p5 is not None:
        spts = sorted([(v,c) for v,c in [(s24p1,10),(s24p5,50),(s24p9,90)] if v is not None])

        def s24_above(thr):
            if not spts: return 0.0
            if thr <= spts[0][0]: return round(100.0 - spts[0][1], 1)
            if thr > spts[-1][0]: return 0.0  # threshold exceeds P90
            for i in range(len(spts)-1):
                v0,c0 = spts[i]; v1,c1 = spts[i+1]
                if v0 <= thr <= v1:
                    pct = c0 + (thr-v0)*(c1-c0)/(v1-v0)
                    return round(max(0.0, 100.0 - pct), 1)
            return 0.0

        # Level thresholds (24hr accumulation):
        #   L2 (MINOR):   > 0" trace
        #   L3 (MODERATE): >= 2" (advisory)
        #   L4 (MAJOR):   >= 4" (warning 12hr) / 6" (24hr) — use 4" as proxy
        #   L5 (EXTREME): >= 8" (well above warning)
        sp, sl, srk = 0.0, 0, 0
        for thr, lvl in [(8.0,5),(4.0,4),(2.0,3),(0.01,2)]:
            p  = s24_above(thr)
            rk = risk_matrix(p, lvl)
            if rk > srk or (rk == srk and p > sp):
                sp, sl, srk = p, lvl, rk

        if srk >= 2:
            # Find peak snow block (highest S01)
            pk_snow_idx = max(
                [i for i in range(len(blocks)) if blocks[i]],
                key=lambda i: (block_hazards[i]['SNOW']['risk'],
                               blocks[i].get('S01', 0))
            )
            pk_blk = blocks[pk_snow_idx]
            threats['SNOW'].update({
                "prob": sp, "risk": srk, "risk_label": RISK_L[srk],
                "color": RISK_C[srk], "level": sl,
                "metric": METRICS["SNOW"].get(sl, ""),
                "s24_p50_in": s24p5,
                **({"peak_start_fxx": pk_blk["start_fxx"],
                    "peak_end_fxx":   pk_blk["end_fxx"],
                    "peak_utc_start": pk_blk["utc_start"]} if pk_blk else {})
            })

    # ───── TEMPERATURE OVERRIDE WITH 24H MAX/MIN ─────────────────────────
    # Use P90 for heat, P1 for cold. Only override if risk >= 2 (MINOR+).
    tmax_p9 = nbp.get('TMAX_D1_P9')
    tmin_p1 = nbp.get('TMIN_D1_P1')
    tmax_p5 = nbp.get('TMAX_D1_P5')
    tmin_p5 = nbp.get('TMIN_D1_P5')
    tmax_p9 = nbp.get('TMAX_D1_P9')

    # HEAT: compute P(temp > threshold) for each impact level, use risk matrix
    if tmax_p5 is not None:
        tpts = sorted([(v,c) for v,c in [
            (nbp.get('TMAX_D1_P1'),10),(tmax_p5,50),(tmax_p9,90)] if v is not None])

        def tmax_above(thr):
            if not tpts: return 0.0
            if thr <= tpts[0][0]: return round(100.0 - tpts[0][1], 1)
            if thr > tpts[-1][0]: return 0.0  # threshold exceeds P90
            for i in range(len(tpts)-1):
                v0,c0 = tpts[i]; v1,c1 = tpts[i+1]
                if v0 <= thr <= v1:
                    pct = c0 + (thr-v0)*(c1-c0)/(v1-v0)
                    return round(max(0.0, 100.0 - pct), 1)
            return 0.0

        hp, hl, h_rk = 0, 0, 0
        for thr, lvl in [(105,5),(100,4),(95,3),(90,2)]:
            p  = tmax_above(thr)
            rk = risk_matrix(p, lvl)
            if rk > h_rk or (rk == h_rk and p > hp):
                hp, hl, h_rk = p, lvl, rk
        if h_rk >= 2:
            pk_blk = blocks[peak_max_block_idx] if peak_max_block_idx < len(blocks) and blocks[peak_max_block_idx] else None
            threats['TEMPERATURE'].update({
                "prob": hp, "risk": h_rk, "risk_label": RISK_L[h_rk],
                "color": RISK_C[h_rk], "level": hl,
                "metric": METRICS["HEAT"].get(hl, ""),
                "temp_type": "heat", "txn_24h_max": tmax_p5,
                **({"peak_start_fxx": pk_blk["start_fxx"],
                    "peak_end_fxx":   pk_blk["end_fxx"],
                    "peak_utc_start": pk_blk["utc_start"]} if pk_blk else {})
            })
            if 0 <= peak_max_block_idx < len(block_hazards) and block_hazards[peak_max_block_idx]:
                block_hazards[peak_max_block_idx]['TEMPERATURE'] = {
                    "prob": hp, "risk": h_rk, "level": hl, "color": RISK_C[h_rk], "temp_type": "heat"
                }

    # COLD: compute P(temp < threshold) for each impact level, use risk matrix
    if tmin_p5 is not None:
        tnpts = sorted([(v,c) for v,c in [
            (tmin_p1,10),(tmin_p5,50),(nbp.get('TMIN_D1_P9'),90)] if v is not None])

        def tmin_below(thr):
            if not tnpts: return 0.0
            if thr >= tnpts[-1][0]: return round(tnpts[-1][1], 1)
            if thr < tnpts[0][0]: return 0.0  # threshold below P1 — essentially 0% chance
            for i in range(len(tnpts)-1):
                v0,c0 = tnpts[i]; v1,c1 = tnpts[i+1]
                if v0 <= thr <= v1:
                    pct = c0 + (thr-v0)*(c1-c0)/(v1-v0)
                    return round(max(0.0, pct), 1)
            return 0.0

        cp, cl, c_rk = 0, 0, 0
        for thr, lvl in [(10,5),(20,4),(32,3),(40,2)]:
            p  = tmin_below(thr)
            rk = risk_matrix(p, lvl)
            if rk > c_rk or (rk == c_rk and p > cp):
                cp, cl, c_rk = p, lvl, rk
        if c_rk >= 2:
            pk_blk = blocks[peak_min_block_idx] if peak_min_block_idx < len(blocks) and blocks[peak_min_block_idx] else None
            threats['TEMPERATURE'].update({
                "prob": cp, "risk": c_rk, "risk_label": RISK_L[c_rk],
                "color": RISK_C[c_rk], "level": cl,
                "metric": METRICS["COLD"].get(cl, ""),
                "temp_type": "cold", "txn_24h_min": tmin_p5,
                **({"peak_start_fxx": pk_blk["start_fxx"],
                    "peak_end_fxx":   pk_blk["end_fxx"],
                    "peak_utc_start": pk_blk["utc_start"]} if pk_blk else {})
            })
            if 0 <= peak_min_block_idx < len(block_hazards) and block_hazards[peak_min_block_idx]:
                block_hazards[peak_min_block_idx]['TEMPERATURE'] = {
                    "prob": cp, "risk": c_rk, "level": cl, "color": RISK_C[c_rk], "temp_type": "cold"
                }



    # ───── DAY 2 FALLBACK FOR THREAT MATRIX ─────────────────────────────────
    # If day 1 shows no threat for WIND, SNOW, or TEMPERATURE, check day 2 NBP.
    # Use the same NOAA workflow: percentile interpolation -> risk matrix.
    # Peak block timing comes from day 2 blocks (indices 8-15).

    def _pct_above(pts, thr):
        if not pts: return 0.0
        if thr <= pts[0][0]: return round(100.0 - pts[0][1], 1)
        if thr > pts[-1][0]: return 0.0  # threshold exceeds P90 — essentially 0%
        for i in range(len(pts)-1):
            v0,c0 = pts[i]; v1,c1 = pts[i+1]
            if v0 <= thr <= v1:
                pct = c0 + (thr-v0)*(c1-c0)/(v1-v0)
                return round(max(0.0, 100.0 - pct), 1)
        return 0.0

    def _pct_below(pts, thr):
        if not pts: return 0.0
        if thr >= pts[-1][0]: return round(pts[-1][1], 1)
        if thr < pts[0][0]: return 0.0  # threshold below P1 — essentially 0%
        for i in range(len(pts)-1):
            v0,c0 = pts[i]; v1,c1 = pts[i+1]
            if v0 <= thr <= v1:
                pct = c0 + (thr-v0)*(c1-c0)/(v1-v0)
                return round(max(0.0, pct), 1)
        return 0.0

    # Find day 2 peak blocks (indices 8-15)
    d2_blocks = blocks[8:16]
    d2_valid  = [b for b in d2_blocks if b]
    d2_pk_gust_blk = max(d2_valid, key=lambda b: b.get('GST',0)) if d2_valid else None
    d2_pk_max_blk  = max(d2_valid, key=lambda b: b.get('TMP',0)) if d2_valid else None
    d2_pk_min_blk  = min(d2_valid, key=lambda b: b.get('TMP',999)) if d2_valid else None

    # SNOW day 2 — always compute and take the higher of day 1 vs day 2
    d2s1 = nbp.get('S24_D2_P1'); d2s5 = nbp.get('S24_D2_P5'); d2s9 = nbp.get('S24_D2_P9')
    if d2s5 is not None:
        d2spts = sorted([(v,c) for v,c in [(d2s1,10),(d2s5,50),(d2s9,90)] if v is not None])
        sp, sl, srk = 0.0, 0, 0
        for thr, lvl in [(8.0,5),(4.0,4),(2.0,3),(0.01,2)]:
            p  = _pct_above(d2spts, thr)
            rk = risk_matrix(p, lvl)
            if rk > srk or (rk == srk and p > sp):
                sp, sl, srk = p, lvl, rk
        if srk >= 2 and srk > threats['SNOW']['risk']:  # Only upgrade if day 2 is higher
            d2_pk_snow_blk = max(d2_valid, key=lambda b: b.get('S01', 0)) if d2_valid else None
            threats['SNOW'].update({
                "prob": sp, "risk": srk, "risk_label": RISK_L[srk],
                "color": RISK_C[srk], "level": sl,
                "metric": METRICS["SNOW"].get(sl, ""),
                "s24_p50_in": d2s5, "day": 2,
                **({"peak_start_fxx": d2_pk_snow_blk["start_fxx"],
                    "peak_end_fxx":   d2_pk_snow_blk["end_fxx"],
                    "peak_utc_start": d2_pk_snow_blk["utc_start"]} if d2_pk_snow_blk else {})
            })

    # WIND day 2 — always compute and take the higher
    d2g1 = nbp.get('G24_D2_P1'); d2g5 = nbp.get('G24_D2_P5'); d2g9 = nbp.get('G24_D2_P9')
    if d2g5 is not None:
        d2pts = sorted([(v,c) for v,c in [(d2g1,10),(d2g5,50),(d2g9,90)] if v is not None])
        bp, bl, brk = 0.0, 0, 0
        for thr, lvl in [(65,5),(58,4),(45,3),(30,2)]:
            p = _pct_above(d2pts, thr)
            rk = risk_matrix(p, lvl)
            if rk > brk or (rk == brk and p > bp):
                bp, bl, brk = p, lvl, rk
        if brk >= 2 and brk > threats['WIND']['risk']:  # Only upgrade if day 2 is higher
            threats['WIND'].update({
                "prob": bp, "risk": brk, "risk_label": RISK_L[brk],
                "color": RISK_C[brk], "level": bl,
                "metric": METRICS["WIND"].get(bl, ""),
                "g24_p10_mph": d2g1, "g24_p50_mph": d2g5, "g24_p90_mph": d2g9,
                "day": 2,
                **({"peak_start_fxx": d2_pk_gust_blk["start_fxx"],
                    "peak_end_fxx":   d2_pk_gust_blk["end_fxx"],
                    "peak_utc_start": d2_pk_gust_blk["utc_start"]} if d2_pk_gust_blk else {})
            })

    # TEMPERATURE day 2 heat — always compute and take the higher
    d2tx5 = nbp.get('TMAX_D2_P5'); d2tx1 = nbp.get('TMAX_D2_P1'); d2tx9 = nbp.get('TMAX_D2_P9')
    if d2tx5 is not None:
        d2tpts = sorted([(v,c) for v,c in [(d2tx1,10),(d2tx5,50),(d2tx9,90)] if v is not None])
        hp, hl, h_rk = 0, 0, 0
        for thr, lvl in [(105,5),(100,4),(95,3),(90,2)]:
            p = _pct_above(d2tpts, thr)
            rk = risk_matrix(p, lvl)
            if rk > h_rk or (rk == h_rk and p > hp):
                hp, hl, h_rk = p, lvl, rk
        if h_rk >= 2 and h_rk > threats['TEMPERATURE']['risk']:
            threats['TEMPERATURE'].update({
                "prob": hp, "risk": h_rk, "risk_label": RISK_L[h_rk],
                "color": RISK_C[h_rk], "level": hl,
                "metric": METRICS["HEAT"].get(hl, ""),
                "temp_type": "heat", "txn_24h_max": d2tx5, "day": 2,
                **({"peak_start_fxx": d2_pk_max_blk["start_fxx"],
                    "peak_end_fxx":   d2_pk_max_blk["end_fxx"],
                    "peak_utc_start": d2_pk_max_blk["utc_start"]} if d2_pk_max_blk else {})
            })

    # TEMPERATURE day 2 cold — always compute and take the higher
    d2tn5 = nbp.get('TMIN_D2_P5'); d2tn1 = nbp.get('TMIN_D2_P1'); d2tn9 = nbp.get('TMIN_D2_P9')
    if d2tn5 is not None:
        d2tnpts = sorted([(v,c) for v,c in [(d2tn1,10),(d2tn5,50),(d2tn9,90)] if v is not None])
        cp, cl, c_rk = 0, 0, 0
        for thr, lvl in [(10,5),(20,4),(32,3),(40,2)]:
            p = _pct_below(d2tnpts, thr)
            rk = risk_matrix(p, lvl)
            if rk > c_rk or (rk == c_rk and p > cp):
                cp, cl, c_rk = p, lvl, rk
        if c_rk >= 2 and c_rk > threats['TEMPERATURE']['risk']:
            threats['TEMPERATURE'].update({
                "prob": cp, "risk": c_rk, "risk_label": RISK_L[c_rk],
                "color": RISK_C[c_rk], "level": cl,
                "metric": METRICS["COLD"].get(cl, ""),
                "temp_type": "cold", "txn_24h_min": d2tn5, "day": 2,
                **({"peak_start_fxx": d2_pk_min_blk["start_fxx"],
                    "peak_end_fxx":   d2_pk_min_blk["end_fxx"],
                    "peak_utc_start": d2_pk_min_blk["utc_start"]} if d2_pk_min_blk else {})
            })
    for i, bh in enumerate(block_hazards):
        if bh and bh.get('TEMPERATURE', {}).get('risk', 0) < 2:
            bh['TEMPERATURE'] = {"prob": 0.0, "risk": 0, "level": 0, "color": RISK_C[0]}

    # Note: WIND blocks with risk=1 (L-T-N) are kept so timeline shows grey blocks
    # during periods where wind is a possibility but not yet at MINOR threshold.

    nbh_hourly = []
    for fxx in range(1, 26):
        h = nbh.get(fxx, {})
        def _kt(v): return round((v or 0)*KT_TO_MPH, 1)
        # At the peak gust hour, replace GST with G24P5 so tooltip shows P50 not mean
        gst_val = _kt(h.get('GST'))
        nbh_hourly.append({
            'fxx': fxx, 'utc': h.get('utc_hour'), 'TMP': h.get('TMP'), 'TSD': h.get('TSD'),
            'DPT': h.get('DPT'), 'WDR': h.get('WDR'), 'WSP': _kt(h.get('WSP')), 'GST': gst_val,
            'GSD': _kt(h.get('GSD')), 'SKY': h.get('SKY'), 'T01': h.get('T01'), 'P01': h.get('P01'),
            'Q01': h.get('Q01'), 'VIS': h.get('VIS'), 'LIV': h.get('LIV'), 'IFV': h.get('IFV'), 'MVV': h.get('MVV')
        })
    with open('threats.json', 'w') as f: json.dump({
        "threats": threats,
        "cycle_utc_iso": f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}T{hs}:00:00Z",
        "cycle": f"NBH {hs}Z",
        "tmax_d1_p5": nbp.get('TMAX_D1_P5'),
        "tmin_d1_p5": nbp.get('TMIN_D1_P5'),
        "tmax_d2_p5": nbp.get('TMAX_D2_P5'),
        "tmin_d2_p5": nbp.get('TMIN_D2_P5'),
    }, f)
    with open('timeline.json', 'w') as f: json.dump({"blocks": blocks, "block_hazards": block_hazards, "nbh_hourly": nbh_hourly}, f)

if __name__ == "__main__": main()
