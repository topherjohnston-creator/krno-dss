"""
downloader.py
Fetches REFS GRIB2 data from AWS S3 using byte-range requests.
Only downloads the specific variables we need — avoids pulling full 500MB+ files.
"""

import requests
import tempfile
import os
import logging
from datetime import datetime, timezone, timedelta

log = logging.getLogger(__name__)

S3_BASE = "https://noaa-rrfs-pds.s3.amazonaws.com"
MEMBERS = ['m001', 'm002', 'm003', 'm004', 'm005']

# Variables we need per forecast hour
# Format: (grib_shortname, level_string_in_idx)
SURFACE_VARS = [
    ('GUST',  'surface'),         # Wind gust (m/s)
    ('VIS',   'surface'),         # Visibility (m)
    ('TMP',   '2 m above ground'),# 2m Temperature (K)
    ('DPT',   '2 m above ground'),# 2m Dew Point (K)
    ('PRATE', 'surface'),         # Precipitation rate (kg/m2/s)
    ('APCP',  'surface'),         # Accum precip (kg/m2)
    ('CPOFP', 'surface'),         # % frozen precip
    ('LTNG',  'surface'),         # Lightning (flashes/km2/day)
    ('WEASD', 'surface'),         # Snow water equiv (kg/m2)
]


def get_latest_cycle():
    """
    Find the most recent available REFS cycle.
    REFS runs at 00/06/12/18 UTC.
    Returns (date_str, cycle_hour_str) e.g. ('20260507', '12')
    """
    now = datetime.now(timezone.utc)
    # Try cycles going back up to 24 hours
    for hours_back in range(0, 25, 6):
        t = now - timedelta(hours=hours_back)
        cycle_hour = (t.hour // 6) * 6
        t_cycle = t.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        date_str = t_cycle.strftime('%Y%m%d')
        hour_str = f"{cycle_hour:02d}"

        # Check if this cycle exists by probing member 1 forecast hour 1
        probe_url = f"{S3_BASE}/rrfs_a/rrfsens.{date_str}/{hour_str}/m001/"
        try:
            r = requests.head(probe_url, timeout=5)
            if r.status_code in (200, 403):  # 403 = exists but no list permission
                log.info(f"Found REFS cycle: {date_str}/{hour_str}Z")
                return date_str, hour_str
        except Exception:
            continue

    raise RuntimeError("Could not find a recent REFS cycle on AWS S3")


def build_member_url(date_str, hour_str, member, fxx):
    """
    Build the S3 URL for a REFS member surface file.
    Tries multiple known filename patterns.
    """
    fxx_str = f"{fxx:03d}"
    hour_str_pad = hour_str.zfill(2)

    # Known filename patterns for RRFS ensemble surface/2D fields
    patterns = [
        f"rrfs.t{hour_str_pad}z.prslev.f{fxx_str}.conus_3km.grib2",
        f"rrfs.t{hour_str_pad}z.natlev.f{fxx_str}.conus_3km.grib2",
        f"rrfs.t{hour_str_pad}z.2dvaraf.f{fxx_str}.conus_3km.grib2",
        f"rrfs_conus.t{hour_str_pad}z.wrfsfcf{fxx_str}.grib2",
    ]

    base_path = f"{S3_BASE}/rrfs_a/rrfsens.{date_str}/{hour_str}/{member}/"
    return [(base_path + p, base_path + p + ".idx") for p in patterns]


def fetch_idx(idx_url):
    """Download and parse a .idx file. Returns list of (varname, level, offset, end_offset)."""
    r = requests.get(idx_url, timeout=15)
    if r.status_code != 200:
        return None

    entries = []
    lines = r.text.strip().split('\n')
    for i, line in enumerate(lines):
        parts = line.split(':')
        if len(parts) < 6:
            continue
        try:
            offset = int(parts[1])
            varname = parts[3]
            level = parts[4]
            # End offset is start of next entry (or None for last)
            end_offset = int(lines[i+1].split(':')[1]) - 1 if i + 1 < len(lines) else None
            entries.append({
                'varname': varname,
                'level': level,
                'offset': offset,
                'end_offset': end_offset
            })
        except (ValueError, IndexError):
            continue

    return entries


def fetch_grib2_variable(grib_url, offset, end_offset):
    """
    Byte-range fetch of a single GRIB2 message.
    Returns raw bytes of just that variable.
    """
    headers = {'Range': f'bytes={offset}-{end_offset}' if end_offset else f'bytes={offset}-'}
    r = requests.get(grib_url, headers=headers, timeout=30)
    if r.status_code not in (200, 206):
        raise RuntimeError(f"Byte-range fetch failed: {r.status_code} for {grib_url}")
    return r.content


def download_surface_vars_for_hour(date_str, hour_str, member, fxx):
    """
    Download only the surface variables we need for one member at one forecast hour.
    Returns path to a temp GRIB2 file containing just those variables,
    or None if download fails.
    """
    url_candidates = build_member_url(date_str, hour_str, member, fxx)

    for grib_url, idx_url in url_candidates:
        log.debug(f"Trying: {idx_url}")
        idx_entries = fetch_idx(idx_url)
        if idx_entries is None:
            continue

        # Find the variables we need
        chunks = []
        for var_name, var_level in SURFACE_VARS:
            for entry in idx_entries:
                if entry['varname'] == var_name and var_level.lower() in entry['level'].lower():
                    try:
                        data = fetch_grib2_variable(grib_url, entry['offset'], entry['end_offset'])
                        chunks.append(data)
                        log.debug(f"  Got {var_name}:{var_level} ({len(data)} bytes)")
                    except Exception as e:
                        log.warning(f"  Failed to get {var_name}: {e}")
                    break

        if not chunks:
            log.warning(f"No variables found in {grib_url}")
            continue

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.grib2')
        for chunk in chunks:
            tmp.write(chunk)
        tmp.close()

        log.info(f"Downloaded {len(chunks)}/{len(SURFACE_VARS)} vars for {member} f{fxx:03d}")
        return tmp.name

    log.warning(f"Could not download any data for {member} f{fxx:03d}")
    return None
