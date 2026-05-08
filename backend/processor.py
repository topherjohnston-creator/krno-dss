"""
processor.py
Extracts KRNO point values from REFS GRIB2, computes ensemble probabilities.
"""

import os
import numpy as np
import logging
import cfgrib
from downloader import get_latest_cycle, download_surface_vars_for_hour, MEMBERS
from calculator import compute_hazard_probabilities

log = logging.getLogger(__name__)

KRNO_LAT =  39.4986
KRNO_LON = 360 - 119.7681  # 240.2319 in 0-360

FORECAST_HOURS = list(range(1, 49))


def extract_point_values(grib_path):
    """Open GRIB2 and extract values at nearest KRNO grid point."""
    values = {}
    try:
        datasets = cfgrib.open_datasets(grib_path)
        for ds in datasets:
            lats = ds.latitude.values
            lons = ds.longitude.values
            if lats.ndim == 2:
                dist = (lats - KRNO_LAT)**2 + (lons - KRNO_LON)**2
                iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
            else:
                iy = np.argmin(np.abs(lats - KRNO_LAT))
                ix = np.argmin(np.abs(lons - KRNO_LON))
            for var in ds.data_vars:
                try:
                    arr = ds[var].values
                    val = float(arr[iy, ix]) if arr.ndim == 2 else float(arr.flat[0])
                    values[var] = val
                except Exception:
                    pass
    except Exception as e:
        log.warning(f"cfgrib error: {e}")
    finally:
        try:
            os.unlink(grib_path)
        except Exception:
            pass
    return values


def map_cfgrib_names(raw):
    name_map = {
        'gust': 'GUST', 'vis': 'VIS', 't2m': 'TMP2M', 'd2m': 'DPT2M',
        'prate': 'PRATE', 'tp': 'APCP', 'cpofp': 'CPOFP',
        'ltng': 'LTNG', 'sdwe': 'WEASD', 'weasd': 'WEASD',
    }
    return {name_map.get(k.lower(), k.upper()): v for k, v in raw.items()}


def get_refs_probabilities():
    debug_log = []

    date_str, hour_str, base_path = get_latest_cycle()
    cycle_label = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]} {hour_str}Z"
    debug_log.append(f"Using REFS cycle: {cycle_label}")
    debug_log.append(f"Base path: {base_path}")
    debug_log.append(f"Target file pattern: rrfs.t{hour_str}z.{{member}}.2dfld.3km.f{{FXX}}.conus.grib2")

    member_data = {fxx: {} for fxx in FORECAST_HOURS}

    for member in MEMBERS:
        debug_log.append(f"Processing {member}...")
        success = 0
        for fxx in FORECAST_HOURS:
            grib_path = download_surface_vars_for_hour(base_path, member, hour_str, fxx)
            if grib_path is None:
                continue
            raw = extract_point_values(grib_path)
            if raw:
                member_data[fxx][member] = map_cfgrib_names(raw)
                success += 1
        debug_log.append(f"  {member}: {success}/{len(FORECAST_HOURS)} hours")

    total = sum(len(v) for v in member_data.values())
    debug_log.append(f"Total data points: {total}")

    if total == 0:
        raise RuntimeError("No REFS data extracted. Check S3 paths.")

    threats, timeline = compute_hazard_probabilities(member_data, FORECAST_HOURS)

    return {
        'threats': threats,
        'timeline': timeline,
        'cycle': cycle_label,
        'debug': debug_log
    }
