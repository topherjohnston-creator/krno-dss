"""
calculator.py
Computes hazard probabilities from REFS ensemble member values.
Implements the KRNO DSS threshold table and the Weather Risk Matrix.

Probability of exceedance = (# members exceeding threshold) / (total members)

Risk level mapping (from Weather Risk Matrix):
  >90% → Level 5 (Extreme) or major → extreme crossover
  66-90% → Level 4 (Major)
  33-66% → Level 3 (Moderate)
  10-33% → Level 2 (Minor)
  <10% → Level 1 (Little to None)
  0% → Level 0 (None)
"""

import numpy as np
import logging

log = logging.getLogger(__name__)

N_MEMBERS = 5  # REFS has 5 members

# -----------------------------------------------------------------------
# Unit conversions
# -----------------------------------------------------------------------
def ms_to_mph(v):   return v * 2.23694
def ms_to_kt(v):    return v * 1.94384
def k_to_f(v):      return (v - 273.15) * 9/5 + 32
def m_to_sm(v):     return v * 0.000621371
def kgm2_to_in(v):  return v * 0.0393701  # kg/m2 = mm, convert to inches
def kgm2s_to_inhr(v): return v * 141.732  # kg/m2/s to in/hr


# -----------------------------------------------------------------------
# Wet-bulb temperature (Stull 2011 approximation)
# Input: T in Celsius, RH in percent
# Returns: Tw in Celsius
# -----------------------------------------------------------------------
def stull_wetbulb(t_c, rh):
    tw = (t_c * np.arctan(0.151977 * (rh + 8.313659)**0.5)
          + np.arctan(t_c + rh)
          - np.arctan(rh - 1.676331)
          + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
          - 4.686035)
    return tw


def calc_rh(t_k, td_k):
    """Compute RH from temperature and dewpoint in Kelvin."""
    t_c  = t_k  - 273.15
    td_c = td_k - 273.15
    rh = 100 * np.exp((17.625 * td_c) / (243.04 + td_c)) / \
                np.exp((17.625 * t_c)  / (243.04 + t_c))
    return np.clip(rh, 0, 100)


# -----------------------------------------------------------------------
# Probability of exceedance across members
# -----------------------------------------------------------------------
def prob_exceed(values, threshold, direction='above'):
    """
    values: list of member values (may contain None for missing)
    threshold: numeric threshold
    direction: 'above' (value > threshold) or 'below' (value < threshold)
    Returns: probability 0.0-1.0
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return 0.0
    if direction == 'above':
        count = sum(1 for v in valid if v > threshold)
    else:
        count = sum(1 for v in valid if v < threshold)
    return count / N_MEMBERS


def prob_compound(values_a, threshold_a, direction_a, values_b, threshold_b, direction_b):
    """
    Probability both conditions are true simultaneously (per-member AND logic).
    """
    count = 0
    for i in range(N_MEMBERS):
        a = values_a[i] if i < len(values_a) else None
        b = values_b[i] if i < len(values_b) else None
        if a is None or b is None:
            continue
        cond_a = (a > threshold_a) if direction_a == 'above' else (a < threshold_a)
        cond_b = (b > threshold_b) if direction_b == 'above' else (b < threshold_b)
        if cond_a and cond_b:
            count += 1
    return count / N_MEMBERS


# -----------------------------------------------------------------------
# Risk level from probability
# Follows the Weather Risk Matrix from the IDSS Guide
# -----------------------------------------------------------------------
def risk_level_from_prob(prob_pct):
    """
    Convert probability (0-100) to risk level (0-5).
    Based on the NWS Weather Risk Matrix.
    """
    if prob_pct >= 90: return 5
    if prob_pct >= 66: return 4
    if prob_pct >= 33: return 3
    if prob_pct >= 10: return 2
    if prob_pct >  0:  return 1
    return 0


RISK_COLORS = {
    0: '#3a3a3a',   # None — dark gray
    1: '#6b7280',   # Little to None — gray
    2: '#eab308',   # Minor — yellow
    3: '#f97316',   # Moderate — orange
    4: '#ef4444',   # Major — red
    5: '#a855f7',   # Extreme — purple
}

RISK_LABELS = {0: 'NONE', 1: 'LITTLE TO NONE', 2: 'MINOR', 3: 'MODERATE', 4: 'MAJOR', 5: 'EXTREME'}


# -----------------------------------------------------------------------
# Per-hazard probability computation
# -----------------------------------------------------------------------

def compute_wind_prob(hour_members):
    """Wind gust probabilities. Thresholds: 30/45/58/65 mph → Impact L2/3/4/5"""
    gusts = [m.get('GUST') for m in hour_members]
    gusts_mph = [ms_to_mph(g) if g is not None else None for g in gusts]

    # Highest impact level with probability >0
    for threshold, level in [(65, 5), (58, 4), (45, 3), (30, 2)]:
        p = prob_exceed(gusts_mph, threshold) * 100
        if p > 0:
            return round(p), level
    return 0, 0


def compute_wind_prob_at_level(hour_members, impact_level):
    """Return probability of wind exceeding the threshold for a given impact level."""
    thresholds = {2: 30, 3: 45, 4: 58, 5: 65}
    thresh = thresholds.get(impact_level, 30)
    gusts = [m.get('GUST') for m in hour_members]
    gusts_mph = [ms_to_mph(g) if g is not None else None for g in gusts]
    return round(prob_exceed(gusts_mph, thresh) * 100)


def compute_snow_prob(hour_members, prev_hour_members=None):
    """
    Snow rate probabilities. Thresholds: T-0.5/0.5-1/1-2/>2 in/hr
    Uses WEASD difference between hours for snow rate.
    If prev_hour not available, uses WEASD as proxy.
    """
    if prev_hour_members:
        # Compute hourly snow accumulation as WEASD diff
        rates = []
        for m_curr, m_prev in zip(hour_members, prev_hour_members):
            curr = m_curr.get('WEASD')
            prev = m_prev.get('WEASD')
            if curr is not None and prev is not None:
                rate_in = kgm2_to_in(max(0, curr - prev))
                rates.append(rate_in)
            else:
                rates.append(None)
    else:
        # Fallback: treat WEASD directly as accumulation
        rates = [kgm2_to_in(m.get('WEASD', 0) or 0) for m in hour_members]

    for threshold, level in [(2.0, 5), (1.0, 4), (0.5, 3), (0.1, 2)]:
        p = prob_exceed(rates, threshold) * 100
        if p > 0:
            return round(p), level
    return 0, 0


def compute_flash_freeze_prob(hour_members):
    """
    Flash freeze: wet + cold wet-bulb temperature.
    Conditions: PRATE > 0 AND Tw < threshold.
    Thresholds: 35/32/28/25°F Tw → Impact L2/3/4/5
    """
    probs_by_level = []
    for thresh_f, level in [(35, 2), (32, 3), (28, 4), (25, 5)]:
        thresh_c = (thresh_f - 32) * 5/9
        count = 0
        valid = 0
        for m in hour_members:
            t = m.get('TMP2M')
            td = m.get('DPT2M')
            prate = m.get('PRATE')
            if t is None or td is None or prate is None:
                continue
            valid += 1
            rh = calc_rh(t, td)
            tw = stull_wetbulb(t - 273.15, rh)
            is_wet = prate > 0.0001  # ~0.001 mm/hr threshold for "wet"
            is_cold = tw < thresh_c
            if is_wet and is_cold:
                count += 1
        p = (count / N_MEMBERS) * 100 if valid > 0 else 0
        if p > 0:
            return round(p), level
    return 0, 0


def compute_lightning_prob(hour_members):
    """
    Lightning probabilities. LTNG field is already a probability/intensity.
    Thresholds: 5/25/50/75% chance → Impact L2/3/4/5
    """
    ltng_vals = [m.get('LTNG') for m in hour_members]
    # LTNG in GRIB2 is typically flashes/km2/day or a probability field
    # Treat as probability field (0-1 scale), convert to percent
    ltng_pct = [(v * 100) if v is not None and v <= 1 else v for v in ltng_vals]

    for threshold, level in [(75, 5), (50, 4), (25, 3), (5, 2)]:
        p = prob_exceed(ltng_pct, threshold) * 100
        if p > 0:
            return round(p), level
    return 0, 0


def compute_visibility_prob(hour_members):
    """
    Visibility probabilities. Thresholds: <5/<3/<1/<0.5 SM → Impact L2/3/4/5
    """
    vis_vals = [m.get('VIS') for m in hour_members]
    vis_sm = [m_to_sm(v) if v is not None else None for v in vis_vals]

    for threshold, level in [(0.5, 5), (1.0, 4), (3.0, 3), (5.0, 2)]:
        p = prob_exceed(vis_sm, threshold, direction='below') * 100
        if p > 0:
            return round(p), level
    return 0, 0


def compute_cold_prob(hour_members):
    """Cold temperature. Thresholds: <32/<20/<10/<0°F → Impact L2/3/4/5"""
    temps = [m.get('TMP2M') for m in hour_members]
    temps_f = [k_to_f(t) if t is not None else None for t in temps]

    for threshold, level in [(0, 5), (10, 4), (20, 3), (32, 2)]:
        p = prob_exceed(temps_f, threshold, direction='below') * 100
        if p > 0:
            return round(p), level
    return 0, 0


def compute_heat_prob(hour_members):
    """Heat temperature. Thresholds: >90/>95/>100/>105°F → Impact L2/3/4/5"""
    temps = [m.get('TMP2M') for m in hour_members]
    temps_f = [k_to_f(t) if t is not None else None for t in temps]

    for threshold, level in [(105, 5), (100, 4), (95, 3), (90, 2)]:
        p = prob_exceed(temps_f, threshold) * 100
        if p > 0:
            return round(p), level
    return 0, 0


def compute_rain_prob(hour_members):
    """
    Rain/flooding: hourly liquid precip rate.
    Uses APCP (hourly accum) filtered by CPOFP < 50% (liquid-dominant).
    Thresholds: 0.1/0.25/0.5/1.0 in/hr → Impact L2/3/4/5
    """
    rates = []
    for m in hour_members:
        apcp = m.get('APCP')
        cpofp = m.get('CPOFP', 0) or 0
        if apcp is None:
            rates.append(None)
            continue
        # Only count as rain if <50% frozen
        if cpofp < 50:
            rates.append(kgm2_to_in(apcp))
        else:
            rates.append(0.0)

    for threshold, level in [(1.0, 5), (0.5, 4), (0.25, 3), (0.1, 2)]:
        p = prob_exceed(rates, threshold) * 100
        if p > 0:
            return round(p), level
    return 0, 0


def compute_fzra_prob(hour_members):
    """
    Freezing rain: APCP with CPOFP indicating freezing precip type.
    Thresholds: trace/>0.01/>0.10 in → Impact L2/3/4/5
    Uses CPOFP as proxy for freezing rain fraction.
    """
    fzra_accum = []
    for m in hour_members:
        apcp = m.get('APCP')
        cpofp = m.get('CPOFP', 0) or 0
        if apcp is None:
            fzra_accum.append(None)
            continue
        # Approximate FZRA: total precip * frozen fraction, minus snow contribution
        # This is an approximation — proper FZRA needs categorical ptype
        fzra = kgm2_to_in(apcp) * min(cpofp / 100, 0.5)  # cap at 50% of total
        fzra_accum.append(fzra)

    for threshold, level in [(0.10, 5), (0.01, 4), (0.001, 3), (0.0001, 2)]:
        p = prob_exceed(fzra_accum, threshold) * 100
        if p > 0:
            return round(p), level
    return 0, 0


# -----------------------------------------------------------------------
# COMBINED COLD/HEAT hazard
# Returns whichever of cold or heat is more significant
# -----------------------------------------------------------------------
def compute_coldheat_prob(hour_members):
    cold_p, cold_l = compute_cold_prob(hour_members)
    heat_p, heat_l = compute_heat_prob(hour_members)
    if cold_l >= heat_l:
        return cold_p, cold_l, 'COLD'
    return heat_p, heat_l, 'HEAT'


# -----------------------------------------------------------------------
# Master computation function
# -----------------------------------------------------------------------
HAZARDS = [
    'WIND', 'SNOW', 'FLASH_FREEZE', 'LIGHTNING',
    'VISIBILITY', 'COLD_HEAT', 'RAIN', 'FZRA'
]


def compute_hazard_probabilities(member_data, forecast_hours):
    """
    member_data: dict[fxx][member_id] = {var: value}
    forecast_hours: list of hour integers

    Returns:
        threats: dict for Panel 1 (max prob over 48hr per hazard)
        timeline: dict for Panel 2 (hourly prob per hazard)
    """

    # Build timeline: hourly probability per hazard
    timeline = {h: {} for h in HAZARDS}

    for fxx in forecast_hours:
        hour_data = member_data.get(fxx, {})
        prev_data = member_data.get(fxx - 1, {})

        if not hour_data:
            for h in HAZARDS:
                timeline[h][fxx] = {'prob': 0, 'level': 0, 'color': RISK_COLORS[0]}
            continue

        # Get ordered list of member value dicts
        members = [hour_data.get(m, {}) for m in ['m001', 'm002', 'm003', 'm004', 'm005']]
        prev_members = [prev_data.get(m, {}) for m in ['m001', 'm002', 'm003', 'm004', 'm005']]

        calcs = {
            'WIND':        compute_wind_prob(members),
            'SNOW':        compute_snow_prob(members, prev_members if prev_data else None),
            'FLASH_FREEZE': compute_flash_freeze_prob(members),
            'LIGHTNING':   compute_lightning_prob(members),
            'VISIBILITY':  compute_visibility_prob(members),
            'COLD_HEAT':   compute_coldheat_prob(members)[:2],
            'RAIN':        compute_rain_prob(members),
            'FZRA':        compute_fzra_prob(members),
        }

        for hazard, (prob, level) in calcs.items():
            risk = risk_level_from_prob(prob)
            timeline[hazard][fxx] = {
                'prob': prob,
                'level': level,
                'risk': risk,
                'color': RISK_COLORS[risk]
            }

    # Build threats: max probability over full window per hazard
    threats = {}
    for hazard in HAZARDS:
        hour_probs = [timeline[hazard].get(fxx, {}).get('prob', 0) for fxx in forecast_hours]
        max_prob = max(hour_probs) if hour_probs else 0
        max_fxx = forecast_hours[np.argmax(hour_probs)] if hour_probs else 0

        # Find peak window (when does prob first exceed max/2)
        peak_start = next((fxx for fxx in forecast_hours
                          if timeline[hazard].get(fxx, {}).get('prob', 0) >= max_prob * 0.5), 0)

        risk = risk_level_from_prob(max_prob)
        threats[hazard] = {
            'prob': max_prob,
            'risk': risk,
            'risk_label': RISK_LABELS[risk],
            'color': RISK_COLORS[risk],
            'peak_hour': max_fxx,
            'peak_start': peak_start,
        }

    return threats, timeline
