"""
processor.py
Extracts values at KRNO (39.4986N, 239.768E) from GRIB2 files,
computes ensemble probability of exceedance, returns hazard data.
"""

import os
import numpy as np
import logging
import cfgrib
from downloader import get_latest_cycle, download_surface_vars_for_hour, MEMBERS
from calculator import compute_hazard_probabilities

log = logging.getLogger(__name__)

# KRNO coordinates
KRNO_LAT =  39.4986
KRNO_LON = 360 - 119.7681  # Convert to 0-360: 240.2319

# Forecast hours to process (0-48)
FORECAST_HOURS = list(range(1, 49))


def extract_point_values(grib_path):
    """
    Open a GRIB2 file and extract values at the KRNO grid point.
    Returns dict of {varname: value_in_si_units}
    """
    values = {}

    try:
        # Open all datasets in file
        datasets = cfgrib.open_datasets(grib_path)

        for ds in datasets:
            lats = ds.latitude.values
            lons = ds.longitude.values

            # Find nearest grid point to KRNO
            # Handle both 1D and 2D lat/lon arrays
            if lats.ndim == 2:
                dist = (lats - KRNO_LAT)**2 + (lons - KRNO_LON)**2
                iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
            else:
                # Find nearest lat and lon independently (regular grid)
                iy = np.argmin(np.abs(lats - KRNO_LAT))
                ix = np.argmin(np.abs(lons - KRNO_LON))

            # Extract each variable at that point
            for var in ds.data_vars:
                try:
                    arr = ds[var].values
                    if arr.ndim == 2:
                        val = float(arr[iy, ix])
                    elif arr.ndim == 1:
                        val = float(arr[max(iy, ix)])
                    else:
                        val = float(arr.flat[0])
                    values[var] = val
                except Exception as e:
                    log.debug(f"Could not extract {var}: {e}")

    except Exception as e:
        log.warning(f"cfgrib error reading {grib_path}: {e}")

    finally:
        try:
            os.unlink(grib_path)
        except Exception:
            pass

    return values


def map_cfgrib_names(raw_values):
    """
    Map cfgrib variable names to our standard names.
    cfgrib uses CF conventions (e.g. 'gust' not 'GUST').
    """
    name_map = {
        # cfgrib name   : our standard name
        'gust':       'GUST',
        'vis':        'VIS',
        't2m':        'TMP2M',
        'd2m':        'DPT2M',
        'prate':      'PRATE',
        'tp':         'APCP',    # total precip
        'cpofp':      'CPOFP',
        'ltng':       'LTNG',
        'sdwe':       'WEASD',   # snow depth water equivalent
        'weasd':      'WEASD',
    }
    mapped = {}
    for k, v in raw_values.items():
        standard = name_map.get(k.lower(), k.upper())
        mapped[standard] = v
    return mapped


def get_refs_probabilities():
    """
    Main entry point. Downloads REFS data for all members and hours,
    computes hazard probabilities, returns structured JSON-ready dicts.
    """
    debug_log = []

    # Find latest cycle
    date_str, hour_str = get_latest_cycle()
    cycle_label = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]} {hour_str}Z"
    debug_log.append(f"Using REFS cycle: {cycle_label}")

    # Structure: member_data[fxx][member] = {var: value}
    member_data = {fxx: {} for fxx in FORECAST_HOURS}

    # Download and extract for each member and hour
    for member in MEMBERS:
        debug_log.append(f"Processing {member}...")
        success_count = 0

        for fxx in FORECAST_HOURS:
            grib_path = download_surface_vars_for_hour(date_str, hour_str, member, fxx)
            if grib_path is None:
                continue

            raw = extract_point_values(grib_path)
            if raw:
                mapped = map_cfgrib_names(raw)
                member_data[fxx][member] = mapped
                success_count += 1

        debug_log.append(f"  {member}: {success_count}/{len(FORECAST_HOURS)} hours downloaded")

    # Check we got enough data
    total_points = sum(len(v) for v in member_data.values())
    debug_log.append(f"Total data points: {total_points}")

    if total_points == 0:
        raise RuntimeError("No REFS data could be downloaded. Check AWS connectivity.")

    # Compute probabilities
    threats, timeline = compute_hazard_probabilities(member_data, FORECAST_HOURS)

    return {
        'threats': threats,
        'timeline': timeline,
        'cycle': cycle_label,
        'debug': debug_log
    }
