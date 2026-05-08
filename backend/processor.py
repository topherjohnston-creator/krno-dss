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

# Every hour 1-48
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


def fetch_member_hour(args):
    """Worker function for parallel download. Returns (fxx, member, values_dict)."""
    base_path, member, fxx = args
    grib_path = download_surface_vars_for_hour(base_path, member, fxx)
    if grib_path is None:
        return fxx, member, None
    raw = extract_point_values(grib_path)
    if not raw:
        return fxx, member, None
    return fxx, member, map_cfgrib_names(raw)


def get_refs_probabilities(live_log=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def emit(msg):
        log.info(msg)
        if live_log is not None:
            live_log.append(msg)

    emit("Finding latest REFS cycle...")
    date_str, hour_str, base_path = get_latest_cycle()
    cycle_label = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]} {hour_str}Z"
    emit(f"Cycle: {cycle_label} | Path: {base_path}")
    emit(f"Downloading {len(MEMBERS)} members × {len(FORECAST_HOURS)} hours in parallel...")

    member_data = {fxx: {} for fxx in FORECAST_HOURS}

    # Build all (base_path, member, fxx) tasks
    tasks = [(base_path, member, fxx)
             for member in MEMBERS
             for fxx in FORECAST_HOURS]

    success = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_member_hour, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                fxx, member, values = future.result()
                if values:
                    member_data[fxx][member] = values
                    success += 1
            except Exception as e:
                log.warning(f"Worker error: {e}")

    emit(f"Downloaded {success}/{len(tasks)} member-hours successfully")

    total = sum(len(v) for v in member_data.values())
    emit(f"Total data points: {total}")

    if total == 0:
        raise RuntimeError("No REFS data extracted. Check S3 file names and IDX parsing.")

    emit("Computing hazard probabilities...")
    threats, timeline = compute_hazard_probabilities(member_data, FORECAST_HOURS)
    emit("Done.")

    return {
        'threats': threats,
        'timeline': timeline,
        'cycle': cycle_label,
        'debug': live_log if live_log is not None else []
    }
