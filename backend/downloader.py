"""
downloader.py
Fetches REFS GRIB2 data from AWS S3 using byte-range requests.
Uses S3 list API to discover available cycles and file names.
"""

import requests
import tempfile
import os
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

log = logging.getLogger(__name__)

S3_BASE  = "https://noaa-rrfs-pds.s3.amazonaws.com"
S3_NS    = "http://s3.amazonaws.com/doc/2006-03-01/"
MEMBERS  = ['m001', 'm002', 'm003', 'm004', 'm005']

SURFACE_VARS = [
    ('GUST',  'surface'),
    ('VIS',   'surface'),
    ('TMP',   '2 m above ground'),
    ('DPT',   '2 m above ground'),
    ('PRATE', 'surface'),
    ('APCP',  'surface'),
    ('CPOFP', 'surface'),
    ('LTNG',  'surface'),
    ('WEASD', 'surface'),
]


def s3_list(prefix, max_keys=10):
    """
    List objects in the S3 bucket under a given prefix.
    Returns list of key strings.
    """
    url = f"{S3_BASE}/?prefix={prefix}&max-keys={max_keys}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
        keys = [el.text for el in root.findall(f'{{{S3_NS}}}Contents/{{{S3_NS}}}Key')]
        return keys
    except Exception as e:
        log.warning(f"S3 list failed for prefix {prefix}: {e}")
        return []


def get_latest_cycle():
    """
    Find the most recent available REFS cycle by querying S3 directly.
    Tries both rrfs_a/rrfsens and rrfs_public/refs path structures.
    Returns (date_str, hour_str, base_path_template)
    """
    now = datetime.now(timezone.utc)

    # Try up to 48 hours back in 6-hour steps
    for hours_back in range(0, 49, 6):
        t = now - timedelta(hours=hours_back)
        cycle_hour = (t.hour // 6) * 6
        t_cycle = t.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        date_str = t_cycle.strftime('%Y%m%d')
        hour_str = f"{cycle_hour:02d}"

        # Try path structures in order of preference
        prefixes = [
            f"rrfs_a/rrfsens.{date_str}/{hour_str}/m001/",
            f"rrfs_a/refs.{date_str}/{hour_str}/m001/",
            f"rrfs_public/refs.{date_str}/{hour_str}/m001/",
        ]

        for prefix in prefixes:
            keys = s3_list(prefix, max_keys=3)
            if keys:
                # Extract the base path (everything before m001/)
                base = prefix.split('m001/')[0]
                log.info(f"Found REFS cycle at: {prefix}")
                log.info(f"Sample files: {keys[:2]}")
                return date_str, hour_str, base

    raise RuntimeError(
        f"No REFS data found on AWS S3 going back 48 hours from {now.strftime('%Y-%m-%d %H:%M UTC')}. "
        "REFS may be temporarily unavailable."
    )


def get_member_files(base_path, member, fxx):
    """
    List actual GRIB2 files for a given member and forecast hour.
    Returns (grib_url, idx_url) or (None, None) if not found.
    """
    prefix = f"{base_path}{member}/"
    fxx_str = f"{fxx:03d}"

    # List files in member directory matching forecast hour
    keys = s3_list(prefix, max_keys=50)

    # Find a file matching this forecast hour
    for key in keys:
        filename = key.split('/')[-1]
        # Match files containing the forecast hour string
        if f"f{fxx_str}" in filename and filename.endswith('.grib2') and not filename.endswith('.idx'):
            grib_url = f"{S3_BASE}/{key}"
            idx_url  = f"{S3_BASE}/{key}.idx"
            return grib_url, idx_url

    return None, None


def fetch_idx(idx_url):
    """Download and parse a .idx inventory file."""
    try:
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
                level   = parts[4]
                end_offset = int(lines[i+1].split(':')[1]) - 1 if i + 1 < len(lines) else None
                entries.append({
                    'varname':    varname,
                    'level':      level,
                    'offset':     offset,
                    'end_offset': end_offset
                })
            except (ValueError, IndexError):
                continue
        return entries
    except Exception as e:
        log.warning(f"IDX fetch failed {idx_url}: {e}")
        return None


def fetch_grib2_variable(grib_url, offset, end_offset):
    """Byte-range fetch of a single GRIB2 message."""
    headers = {'Range': f'bytes={offset}-{end_offset}' if end_offset else f'bytes={offset}-'}
    r = requests.get(grib_url, headers=headers, timeout=30)
    if r.status_code not in (200, 206):
        raise RuntimeError(f"Byte-range fetch failed: {r.status_code}")
    return r.content


def download_surface_vars_for_hour(base_path, member, fxx):
    """
    Download only the surface variables we need for one member/hour.
    Returns path to temp GRIB2 file, or None if unavailable.
    """
    grib_url, idx_url = get_member_files(base_path, member, fxx)
    if not grib_url:
        log.debug(f"No file found for {member} f{fxx:03d}")
        return None

    idx_entries = fetch_idx(idx_url)
    if not idx_entries:
        log.warning(f"No IDX for {member} f{fxx:03d}: {idx_url}")
        return None

    chunks = []
    for var_name, var_level in SURFACE_VARS:
        for entry in idx_entries:
            if (entry['varname'] == var_name and
                    var_level.lower() in entry['level'].lower()):
                try:
                    data = fetch_grib2_variable(grib_url, entry['offset'], entry['end_offset'])
                    chunks.append(data)
                    log.debug(f"  Got {var_name}:{var_level} ({len(data)} bytes)")
                except Exception as e:
                    log.warning(f"  Byte-range failed {var_name}: {e}")
                break

    if not chunks:
        log.warning(f"No variables extracted for {member} f{fxx:03d}")
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.grib2')
    for chunk in chunks:
        tmp.write(chunk)
    tmp.close()

    log.info(f"Downloaded {len(chunks)}/{len(SURFACE_VARS)} vars for {member} f{fxx:03d}")
    return tmp.name
