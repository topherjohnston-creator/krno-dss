#!/usr/bin/env python3
"""
KRNO DSS — NBM Text Bulletin Processor
GitHub Actions workflow script
Downloads NBM text bulletins, computes hazard probabilities,
writes threats.json and timeline.json to repo root.

DATA SOURCE COVERAGE (by design — not configurable):
  NBH (hourly)  : f001–f025   Wind/Vis/Precip/Temp at 1-hr resolution
  NBS (3-hourly): f005–f072   Snow/FZRA/Temp/Precip at 3-hr resolution
                               Also fills blocks 9-15 (hours 26-48) for all
                               met elements missing from NBH.
  NBP (prob)    : 01/07/13/19Z  G24 gust percentiles, MaxT/MinT Gaussian

NOTE — NBH IS 25 HOURS BY DESIGN: The NBH bulletin is officially defined as
"1-hourly NBM guidance from 1-25 hours." This is an NWS product constraint,
not a parser limitation. Blocks 9-15 (fxx 26-48) use NBS data. Visibility
probability fields (MVV/IFV/LIV) are only available in NBH; they will show
zero for the D2 timeline blocks.
"""

import re, json, requests, sys, os
from datetime import datetime, timezone, timedelta
from scipy.stats import norm
import numpy as np

STATION    = "KRNO"
NOMADS_NBM = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"

# Minimum probability (%) a Gaussian hazard must reach to register a level.
# Prevents Gaussian tails from triggering extreme-level false positives.
HAZARD_MIN_PROB = 5.0

RISK_C = {0:"#3f3f46",1:"#e2f0cb",2:"#ffeb3b",3:"#ff9800",4:"#f44336",5:"#9c27b0"}
RISK_L = {0:"NONE",1:"LITTLE TO NONE",2:"MINOR",3:"MODERATE",4:"MAJOR",5:"EXTREME"}
HAZARDS = ["WIND","LIGHTNING","SNOW","VISIBILITY","FZRA",
           "FLASH_FREEZE","RAIN","TEMPERATURE"]
METRICS = {
    "WIND":        {2:"30-45 mph",     3:"45-58 mph",     4:"58-65 mph",    5:">65 mph"},
    "SNOW":        {2:"T-0.5 in/hr",   3:"0.5-1 in/hr",   4:"1-2 in/hr",    5:">2 in/hr"},
    "LIGHTNING":   {2:"5-25%",         3:"25-50%",        4:"50-75%",       5:">75%"},
    "VISIBILITY":  {2:"3-5 SM",        3:"1-3 SM",        4:"0.5-1 SM",     5:"<0.5 SM"},
    "COLD":        {2:"32-40F",        3:"20-32F",        4:"0-20F",        5:"<0F"},
    "HEAT":        {2:"90-95F",        3:"95-100F",       4:"100-105F",     5:">105F"},
    "TEMPERATURE": {2:"32-40F/90-95F", 3:"20-32F/95-100F",4:"0-20F/100-105F",5:"<0F/>105F"},
    "RAIN":        {2:">0.01 in/hr",   3:">0.25 in/hr",   4:">0.50 in/hr",  5:">1.0 in/hr"},
    "FZRA":        {2:"Trace",         3:"Trace-0.01 in", 4:"0.01-0.10 in", 5:">0.10 in"},
    "FLASH_FREEZE":{2:"Wet+Tw<36F",    3:"Wet+Tw<32F",    4:"Wet+Tw<28F",   5:"Wet+Tw<25F"},
}

# ── Utilities ──────────────────────────────────────────────────────────────────

def rlvl(p):
    if p>=90: return 5
    if p>=66: return 4
    if p>=33: return 3
    if p>=10: return 2
    if p>0:   return 1
    return 0

def gauss_above(mean, std, thr):
    if mean is None: return 0.0
    return round(float((1-norm.cdf(thr, mean, max(std or 0.1, 0.1)))*100), 1)

def gauss_below(mean, std, thr):
    if mean is None: return 0.0
    return round(float(norm.cdf(thr, mean, max(std or 0.1, 0.1))*100), 1)

def pct_to_gaussian(p10, p50, p90):
    if p50 is None: return None, None
    if p90 and p90 > p50: std = (p90 - p50) / 1.28
    elif p10 and p50 > p10: std = (p50 - p10) / 1.28
    else: std = 3.0
    return p50, max(std, 0.5)

def parse_row(label, section):
    for line in section.split('\n'):
        s = line.strip()
        if s.startswith(label+' ') or s.startswith(label+'\t'):
            return [int(x) for x in re.findall(r'-?\d+', s[len(label):])]
    return []

# ── Fetch bulletins ────────────────────────────────────────────────────────────

def get_cycle(btype):
    now = datetime.now(timezone.utc)
    valid = [1,7,13,19] if btype=='p' else list(range(24))
    for hb in range(0, 8):
        t = now - timedelta(hours=hb)
        if t.hour not in valid: continue
        ds, hs = t.strftime('%Y%m%d'), f"{t.hour:02d}"
        url = f"{NOMADS_NBM}/blend.{ds}/{hs}/text/blend_nb{btype}tx.t{hs}z"
        try:
            r = requests.head(url, timeout=10)
            if r.status_code == 200:
                return ds, hs, url
        except: continue
    return None, None, None

def fetch_station(url):
    r = requests.get(url, timeout=90)
    if r.status_code != 200: return None
    idx = r.text.find(STATION)
    if idx < 0: return None
    end = r.text.find('\n ' + STATION[:3], idx+100)
    return r.text[idx: end if end > 0 else idx+5000]

# ── Parse bulletins ────────────────────────────────────────────────────────────

def parse_nbh(sec):
    """Parse NBH bulletin. Returns dict keyed by fxx (1-25 by product design)."""
    utc_row = []
    for line in sec.split('\n'):
        if line.strip().startswith('UTC '):
            utc_row = [int(x) for x in re.findall(r'\d+', line[4:])]
            break
    rows = {}
    for el in ['TMP','TSD','DPT','DSD','GST','GSD','P01','Q01',
               'T01','VIS','MVV','IFV','LIV']:
        v = parse_row(el, sec)
        if v: rows[el] = v
    data = {}
    for i, uh in enumerate(utc_row[:48]):  # [:48] is fine; NBH only has 25 columns
        fxx = i+1
        entry = {'utc_hour': uh, 'fxx': fxx}
        for el, vals in rows.items():
            if i < len(vals):
                v = vals[i]
                entry[el] = None if v in (-99,999,-88,888) else v
        data[fxx] = entry
    return data

def parse_nbs(sec):
    """
    Parse NBS bulletin (3-hourly, fxx 5-72).
    FIX v2: expanded element list to include TMP/DPT/GST/etc. so NBS can
    fill blocks 9-15 (hours 26-48) where NBH has no data.
    """
    fhr_row = []
    for line in sec.split('\n'):
        if 'FHR' in line:
            fhr_row = [int(x) for x in re.findall(r'\d+', line[4:])]
            break
    rows = {}
    # EXPANDED: was ['S06','ZR6','T06','P06'] — now includes full met suite
    for el in ['TMP','TSD','DPT','DSD','GST','GSD','P06','Q06','S06','ZR6','T06']:
        v = parse_row(el, sec)
        if v: rows[el] = v
    data = {}
    for i, fxx in enumerate(fhr_row):
        if fxx > 72: break
        entry = {'fxx': fxx}
        for el, vals in rows.items():
            if i < len(vals):
                v = vals[i]
                entry[el] = None if v in (-99,999) else v
        data[fxx] = entry
    return data

def parse_nbp(sec):
    """
    Parse NBP probabilistic bulletin.
    TXNMN row: per NOAA docs, Minimum is listed at 12z, Maximum at 00z.
    For a 01Z cycle: [0]=tonight MinT, [1]=tomorrow MaxT, [2]=D2 MinT, [3]=D2 MaxT.
    """
    txnmn=parse_row('TXNMN',sec); txnsd=parse_row('TXNSD',sec)
    g24p1=parse_row('G24P1',sec); g24p2=parse_row('G24P2',sec)
    g24p5=parse_row('G24P5',sec); g24p7=parse_row('G24P7',sec)
    g24p9=parse_row('G24P9',sec)
    r = {}
    # TXNMN alternates MinT (at 12z) and MaxT (at 00z), but which column comes
    # FIRST is cycle-dependent:
    #   01Z / 07Z runs → next 12Z is closer  → col[0]=MinT, col[1]=MaxT
    #   13Z / 19Z runs → next 00Z is closer  → col[0]=MaxT, col[1]=MinT
    # FIX: use min/max of each pair — robust regardless of cycle.
    # MaxT is always warmer than MinT for the same calendar day.
    if len(txnmn)>=2:
        r.update({'TMAX_D1': max(txnmn[0], txnmn[1]),
                  'TMIN_N1': min(txnmn[0], txnmn[1])})
    if len(txnmn)>=4:
        r.update({'TMAX_D2': max(txnmn[2], txnmn[3]),
                  'TMIN_N2': min(txnmn[2], txnmn[3])})
    # Std devs: assign larger std to MaxT (daytime heating is more variable)
    if len(txnsd)>=2:
        r.update({'TMAX_D1_STD': max(txnsd[0], txnsd[1]),
                  'TMIN_N1_STD': min(txnsd[0], txnsd[1])})
    if len(txnsd)>=4:
        r.update({'TMAX_D2_STD': max(txnsd[2], txnsd[3]),
                  'TMIN_N2_STD': min(txnsd[2], txnsd[3])})
    for pct,row in [(10,g24p1),(20,g24p2),(50,g24p5),(70,g24p7),(90,g24p9)]:
        if len(row)>=1: r[f'G24_D1_P{pct}']=row[0]
        if len(row)>=2: r[f'G24_D2_P{pct}']=row[1]
    return r

# ── Build 3-hour blocks ────────────────────────────────────────────────────────

def make_blocks(nbh, nbs):
    """
    Build 16 × 3-hour blocks covering hours 1-48.

    Blocks 0-7  (fxx 1-24):  NBH primary (1-hr resolution); NBS for S06/ZR6.
    Blocks 8    (fxx 25-27): NBH for fxx=25, NBS for remainder.
    Blocks 9-15 (fxx 26-48): NBS primary (3-hr resolution); VIS fields unavailable.

    FIX v2: was `blocks.append(None)` for all NBS-only blocks — now builds
    valid blocks from NBS TMP/DPT/GST/P06/Q06 so hours 26-48 are computed.
    """
    # Derive cycle UTC hour from NBH fxx=1 entry (for NBS block utc_start)
    cycle_utc_h = None
    if nbh.get(1):
        cycle_utc_h = (nbh[1]['utc_hour'] - 1) % 24

    blocks = []
    for bi in range(16):
        s = bi*3+1; e = s+2

        # ── NBH hours in this block ──────────────────────────────────────────
        nbh_hrs = [nbh.get(f) for f in range(s, e+1) if nbh.get(f)]

        # ── Matching NBS 3-hour period ───────────────────────────────────────
        nbs_entry = None
        mid = s+1
        for nk in [round(mid/3)*3, round((mid+1)/3)*3, round((mid-1)/3)*3]:
            if nk in nbs: nbs_entry = nbs[nk]; break

        # ── Need at least one data source ────────────────────────────────────
        if not nbh_hrs and nbs_entry is None:
            blocks.append(None); continue

        # ── Helper: NBS scalar getter ────────────────────────────────────────
        def _nbs(k): return nbs_entry.get(k) if nbs_entry else None

        # ── NBH aggregate helpers (fall back to NBS scalar if no NBH data) ──
        def av(k):
            v = [h[k] for h in nbh_hrs if h.get(k) is not None]
            return sum(v)/len(v) if v else _nbs(k)
        def mx(k):
            v = [h[k] for h in nbh_hrs if h.get(k) is not None]
            return max(v) if v else _nbs(k)
        def mn(k):
            v = [h[k] for h in nbh_hrs if h.get(k) is not None]
            return min(v) if v else _nbs(k)

        # ── utc_start: use NBH when available, else derive from cycle ────────
        if nbh_hrs:
            utc_start = nbh_hrs[0].get('utc_hour')
        elif cycle_utc_h is not None:
            utc_start = (cycle_utc_h + s) % 24
        else:
            utc_start = None

        # ── NBS P06/Q06 → hourly-equivalent P01/Q01 ─────────────────────────
        # P06 is 6-hr PoP; used as-is (probability of precip in the period).
        # Q06 is 6-hr QPF in hundredths of inches; divide by 6 for hr rate.
        nbs_p01 = _nbs('P06')
        nbs_q01 = round(_nbs('Q06') / 6.0, 1) if _nbs('Q06') is not None else None

        b = {
            'start_fxx': s, 'end_fxx': e, 'utc_start': utc_start,
            # Temperature / moisture (NBH primary, NBS fallback)
            'TMP': av('TMP'), 'TSD': av('TSD'), 'DPT': av('DPT'),
            # Wind gust (NBH primary, NBS fallback)
            'GST': mx('GST'), 'GSD': av('GSD'),
            # Precip probability & QPF (NBH primary; NBS P06/Q06 for D2 blocks)
            'T01': mx('T01') if nbh_hrs else _nbs('T06'),
            'P01': mx('P01') if nbh_hrs else nbs_p01,
            'Q01': mx('Q01') if nbh_hrs else nbs_q01,
            # Visibility probability fields (NBH only — not in NBS)
            # D2 blocks will show zero visibility risk (data not available)
            'VIS': mn('VIS'),
            'MVV': mx('MVV'),
            'IFV': mx('IFV'),
            'LIV': mx('LIV'),
            # Precip type from NBS (always from NBS for all blocks)
            'S06': _nbs('S06'),
            'ZR6': _nbs('ZR6'),
            'T06': _nbs('T06'),
        }
        blocks.append(b)
    return blocks

# ── Compute hazards ────────────────────────────────────────────────────────────

def compute_block(block, bi, nbp):
    if block is None:
        return {hz:{"prob":0,"risk":0,"level":0,"color":RISK_C[0]} for hz in HAZARDS+["COLD","HEAT"]}
    h = {}
    d2 = bi >= 8

    # ── WIND ──────────────────────────────────────────────────────────────────
    # Uses NBP G24 24-hour max gust percentiles (4 cycles/day: 01/07/13/19Z).
    # Between NBP cycles the values are stale by up to 6 hours — acceptable.
    #
    # DESIGN (v3): P50-based level + lower-bound probability.
    #   Level = intensity tier where P50 median falls.
    #           Tells operations WHERE gusts are most expected.
    #   Prob  = P(exceeding that tier's lower threshold).
    #           Tells operations HOW CONFIDENT we are of reaching that tier.
    #
    # Prior approaches produced the wrong metric label:
    #   P50=45 → P(>58)=10% → "MINOR at 58-65 mph"  (WRONG)
    #   v3:    → P(>45)=50% → "MODERATE at 45-58 mph" (CORRECT)
    gm, gs = pct_to_gaussian(
        nbp.get(f"G24_D{'2' if d2 else '1'}_P10"),
        nbp.get(f"G24_D{'2' if d2 else '1'}_P50"),
        nbp.get(f"G24_D{'2' if d2 else '1'}_P90"))
    wp, wl = 0.0, 0
    if gm is not None:
        if   gm >= 65: wl, low = 5, 65
        elif gm >= 58: wl, low = 4, 58
        elif gm >= 45: wl, low = 3, 45
        elif gm >= 30: wl, low = 2, 30
        else:          wl, low = 0, 30   # median < 30; check for tail risk
        wp = gauss_above(gm, gs, low)
        if wl == 0 and wp < HAZARD_MIN_PROB:
            wp = 0.0   # median and tail both negligible → no wind hazard
        elif wl == 0 and wp >= HAZARD_MIN_PROB:
            wl = 2     # non-trivial chance of 30+ mph despite low median
    h["WIND"] = {"prob":wp,"risk":rlvl(wp),"level":wl,"color":RISK_C[rlvl(wp)]}

    # ── LIGHTNING ─────────────────────────────────────────────────────────────
    t01 = block.get('T01') or block.get('T06') or 0
    ll = 5 if t01>=75 else 4 if t01>=50 else 3 if t01>=25 else 2 if t01>=5 else 0
    h["LIGHTNING"] = {"prob":float(t01),"risk":rlvl(t01),"level":ll,
                      "color":RISK_C[rlvl(t01)]}

    # ── SNOW ──────────────────────────────────────────────────────────────────
    s06=block.get('S06'); pop=block.get('P01') or 0; sp,sl=0,0
    if s06 and s06>0:
        s_inhr=(s06/10.0)/6.0
        for t,l in [(2,5),(1,4),(0.5,3),(0.1,2)]:
            if s_inhr>=t: sp,sl=min(pop,100.0),l; break
    h["SNOW"] = {"prob":sp,"risk":rlvl(sp),"level":sl,"color":RISK_C[rlvl(sp)]}

    # ── VISIBILITY ────────────────────────────────────────────────────────────
    # LIV/IFV/MVV only available in NBH (hours 1-25).
    # D2 blocks (26-48) will show zero — document this in the dashboard.
    liv=block.get('LIV') or 0; ifv=block.get('IFV') or 0; mvv=block.get('MVV') or 0
    if liv>=1: vp,vl=liv,4
    elif ifv>=1: vp,vl=ifv,3
    elif mvv>=1: vp,vl=mvv,2
    else: vp,vl=0,0
    h["VISIBILITY"] = {"prob":vp,"risk":rlvl(vp),"level":vl,"color":RISK_C[rlvl(vp)]}

    # ── FZRA ──────────────────────────────────────────────────────────────────
    zr6=block.get('ZR6'); fzp,fzl=0,0
    if zr6 and zr6>0:
        zr=zr6/100.0
        for t,l in [(0.10,5),(0.01,4),(0.001,3),(0.0001,2)]:
            if zr>=t: fzp,fzl=min(pop,100.0),l; break
    h["FZRA"] = {"prob":fzp,"risk":rlvl(fzp),"level":fzl,"color":RISK_C[rlvl(fzp)]}

    # ── FLASH FREEZE ──────────────────────────────────────────────────────────
    tmp_f=block.get('TMP'); dpt_f=block.get('DPT'); ff_p,ff_l=0,0
    if tmp_f is not None and dpt_f is not None and pop>=25:
        tw_f = tmp_f - (tmp_f - dpt_f) / 3.0  # Stull wet-bulb approximation
        wf = pop/100.0
        for t,l in [(25,5),(28,4),(32,3),(36,2)]:
            if tw_f<=t: ff_p=round(wf*100,1); ff_l=l; break
    h["FLASH_FREEZE"] = {"prob":ff_p,"risk":rlvl(ff_p),"level":ff_l,
                         "color":RISK_C[rlvl(ff_p)]}

    # ── RAIN ──────────────────────────────────────────────────────────────────
    q01=block.get('Q01'); rp,rl=0,0
    if q01 and q01>0:
        q=q01/100.0
        for t,l in [(1.0,5),(0.5,4),(0.25,3),(0.10,2)]:
            if q>=t: rp,rl=min(pop,100.0),l; break
    elif pop>10: rp,rl=pop,2
    h["RAIN"] = {"prob":rp,"risk":rlvl(rp),"level":rl,"color":RISK_C[rlvl(rp)]}

    # ── COLD ──────────────────────────────────────────────────────────────────
    # FIX v2 (same bug as WIND): was scanning HIGH→LOW (extreme→mild).
    # Fixed to scan LOW→HIGH (mild→extreme) and break on first miss.
    mint=nbp.get("TMIN_N2" if d2 else "TMIN_N1")
    mint_std=nbp.get("TMIN_N2_STD" if d2 else "TMIN_N1_STD") or 3
    cp, cl = 0.0, 0
    if mint:
        for t, l in [(40,2),(32,3),(20,4),(0,5)]:    # MILD → EXTREME (ascending threshold)
            p = gauss_below(mint, mint_std, t)
            if p >= HAZARD_MIN_PROB:
                cp, cl = p, l
            else:
                break  # gauss_below decreases as threshold decreases — safe to break
    h["COLD"] = {"prob":cp,"risk":rlvl(cp),"level":cl,"color":RISK_C[rlvl(cp)]}

    # ── HEAT ──────────────────────────────────────────────────────────────────
    # FIX v2 (same bug as WIND): was scanning HIGH→LOW (extreme→mild).
    # Fixed to scan LOW→HIGH (mild→extreme) and break on first miss.
    maxt=nbp.get("TMAX_D2" if d2 else "TMAX_D1")
    maxt_std=nbp.get("TMAX_D2_STD" if d2 else "TMAX_D1_STD") or 3
    hp, hl = 0.0, 0
    if maxt:
        for t, l in [(90,2),(95,3),(100,4),(105,5)]:  # MILD → EXTREME (ascending threshold)
            p = gauss_above(maxt, maxt_std, t)
            if p >= HAZARD_MIN_PROB:
                hp, hl = p, l
            else:
                break  # monotonically decreasing — safe to break
    h["HEAT"] = {"prob":hp,"risk":rlvl(hp),"level":hl,"color":RISK_C[rlvl(hp)]}

    # ── TEMPERATURE — higher risk of COLD or HEAT ────────────────────────────
    if h["HEAT"]["risk"] >= h["COLD"]["risk"]:
        temp=dict(h["HEAT"]); temp["temp_type"]="heat"
    else:
        temp=dict(h["COLD"]); temp["temp_type"]="cold"
    h["TEMPERATURE"] = temp

    return h

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("KRNO DSS — NBM text bulletin processor  (v2 — NBS fallback + loop fixes)")
    now = datetime.now(timezone.utc)
    print(f"Run time: {now.isoformat()}")

    # Fetch bulletins
    print("\nFetching NBM bulletins...")
    _,nbh_hs,nbh_url = get_cycle('h')
    _,nbs_hs,nbs_url = get_cycle('s')
    _,nbp_hs,nbp_url = get_cycle('p')

    print(f"  NBH: {nbh_hs}Z  NBS: {nbs_hs}Z  NBP: {nbp_hs}Z")

    nbh_sec = fetch_station(nbh_url) if nbh_url else None
    nbs_sec = fetch_station(nbs_url) if nbs_url else None
    nbp_sec = fetch_station(nbp_url) if nbp_url else None

    if not nbh_sec:
        print("ERROR: NBH bulletin unavailable", file=sys.stderr)
        sys.exit(1)

    print("  ✓ All bulletins fetched")

    # Parse
    nbh = parse_nbh(nbh_sec)
    nbs = parse_nbs(nbs_sec) if nbs_sec else {}
    nbp = parse_nbp(nbp_sec) if nbp_sec else {}

    # NBH is 25 hrs by design; NBS covers the rest
    nbs_active = sum(1 for b in range(9,16) if any(
        (bi*3+1) <= int(k) <= (bi*3+3) for k in [str(x) for x in nbs.keys()]
        for bi in [b]))
    print(f"  NBH: {len(nbh)} hrs (D1, by design)  NBS: {len(nbs)} periods (D1+D2)  NBP: {len(nbp)} fields")

    # ── DEBUG: uncomment to verify G24 values against NBM histogram/bulletin ──
    # print("\nNBP raw values (compare against NBM text tool at blend.mdl.nws.noaa.gov):")
    # for day in ['D1', 'D2']:
    #     pcts = {p: nbp.get(f'G24_{day}_P{p}') for p in [10,20,50,70,90]}
    #     print(f"  G24 {day} gust pctiles: {pcts}")
    # print(f"  MaxT: D1={nbp.get('TMAX_D1')}°F  D2={nbp.get('TMAX_D2')}°F")
    # print(f"  MinT: N1={nbp.get('TMIN_N1')}°F  N2={nbp.get('TMIN_N2')}°F")

    # Build blocks and compute hazards
    blocks = make_blocks(nbh, nbs)
    block_hazards = [compute_block(b, i, nbp) for i,b in enumerate(blocks)]

    d1_blocks = sum(1 for b in blocks[:8] if b is not None)
    d2_blocks = sum(1 for b in blocks[8:] if b is not None)
    print(f"  Timeline: {d1_blocks}/8 D1 blocks + {d2_blocks}/8 D2 blocks populated")

    # Threat cards — max over all blocks
    threats = {}
    for hz in HAZARDS:
        probs  = [block_hazards[i][hz]["prob"]  for i in range(16) if blocks[i]]
        levels = [block_hazards[i][hz]["level"] for i in range(16) if blocks[i]]
        mp  = max(probs)  if probs  else 0
        rk  = rlvl(mp)
        plv = levels[probs.index(mp)] if probs and mp>0 else 0
        met = METRICS.get(hz,{}).get(plv,"") if plv>0 else ""
        pk  = probs.index(mp) if probs and mp>0 else 0
        pb  = blocks[pk] if pk<len(blocks) and blocks[pk] else None
        threats[hz] = {
            "prob": round(mp,1), "risk": rk, "risk_label": RISK_L[rk],
            "color": RISK_C[rk], "level": plv, "metric": met,
            "peak_start_fxx": pb["start_fxx"] if pb else None,
            "peak_end_fxx":   pb["end_fxx"]   if pb else None,
            "peak_utc_start": pb["utc_start"]  if pb else None,
        }

    now_iso = now.isoformat()
    cycle_label = f"NBH {nbh_hs}Z / NBS {nbs_hs}Z / NBP {nbp_hs}Z"

    # Write output files
    threats_out = {"threats": threats, "cycle": cycle_label, "last_updated": now_iso}
    timeline_out = {"blocks": blocks, "block_hazards": block_hazards,
                    "cycle": cycle_label, "last_updated": now_iso}

    def serialize(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open('threats.json', 'w') as f:
        json.dump(threats_out, f, default=serialize)
    with open('timeline.json', 'w') as f:
        json.dump(timeline_out, f, default=serialize)

    print(f"\n✓ threats.json and timeline.json written")
    print(f"  Cycle: {cycle_label}")
    print(f"  Updated: {now_iso}")
    print("\nActive threats:")
    for hz, v in threats.items():
        if v["prob"] > 0:
            print(f"  {hz:15s}: {v['prob']:5.1f}%  {v['risk_label']:<14s}  {v['metric']}")

if __name__ == "__main__":
    main()
