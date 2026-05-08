"""
downloader.py
Fetches REFS GRIB2 data from AWS S3 using byte-range requests.
File naming convention confirmed from S3 inventory:
  rrfs.t{HH}z.{member}.2dfld.3km.f{FXX}.conus.grib2
"""

import requests
import tempfile
import os
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

log = logging.getLogger(__name__)

S3_BASE = "https://noaa-rrfs-pds.s3.amazonaws.com"
S3_NS   = "http://s3.amazonaws.com/doc/2006-03-01/"
MEMBERS = ['m001', 'm002', 'm003', 'm004', 'm005']

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


def s3_list(prefix, max_keys=200):
    """List objects in S3 bucket under a prefix. Returns list of key strings."""
    url = f"{S3_BASE}/?prefix={prefix}&max-keys={max_keys}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
        return [el.text for el in root.findall(f'{{{S3_NS}}}Contents/{{{S3_NS}}}Key')]
    except Exception as e:
        log.warning(f"S3 list failed for {prefix}: {e}")
        return []


def get_latest_cycle():
    """
    Find the most recent available REFS cycle.
    Returns (date_str, hour_str, base_path)
    """
    now = datetime.now(timezone.utc)

    for hours_back in range(0, 49, 6):
        t = now - timedelta(hours=hours_back)
        cycle_hour = (t.hour // 6) * 6
        t_cycle = t.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        date_str = t_cycle.strftime('%Y%m%d')
        hour_str = f"{cycle_hour:02d}"

        prefixes = [
            f"rrfs_a/rrfsens.{date_str}/{hour_str}/m001/",
            f"rrfs_a/refs.{date_str}/{hour_str}/m001/",
            f"rrfs_public/refs.{date_str}/{hour_str}/m001/",
        ]

        for prefix in prefixes:
            keys = s3_list(prefix, max_keys=5)
            if keys:
                base = prefix.split('m001/')[0]
                log.info(f"Found REFS cycle at: {prefix}")
                return date_str, hour_str, base

    raise RuntimeError(
        f"No REFS data found on AWS S3 in the past 48 hours from "
        f"{now.strftime('%Y-%m-%d %H:%M UTC')}."
    )


def build_file_url(base_path, member, hour_str, fxx):
    """
    Build the exact GRIB2 and IDX URLs for a member/forecast hour.
    Confirmed naming pattern: rrfs.t{HH}z.{member}.2dfld.3km.f{FXX}.conus.grib2
    """
    fxx_str = f"{fxx:03d}"
    filename = f"rrfs.t{hour_str}z.{member}.2dfld.3km.f{fxx_str}.conus.grib2"
    grib_url = f"{S3_BASE}/{base_path}{member}/{filename}"
    idx_url  = f"{S3_BASE}/{base_path}{member}/{filename}.idx"
    return grib_url, idx_url


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
                offset     = int(parts[1])
                varname    = parts[3]
                level      = parts[4]
                end_offset = int(lines[i+1].split(':')[1]) - 1 if i+1 < len(lines) else None
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
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.content


def download_surface_vars_for_hour(base_path, member, hour_str, fxx):
    """
    Download only the surface variables we need for one member/hour.
    Returns path to temp GRIB2 file, or None if unavailable.
    """
    grib_url, idx_url = build_file_url(base_path, member, hour_str, fxx)

    idx_entries = fetch_idx(idx_url)
    if not idx_entries:
        log.debug(f"No IDX for {member} f{fxx:03d}: {idx_url}")
        return None

    chunks = []
    for var_name, var_level in SURFACE_VARS:
        for entry in idx_entries:
            if (entry['varname'] == var_name and
                    var_level.lower() in entry['level'].lower()):
                try:
                    data = fetch_grib2_variable(grib_url, entry['offset'], entry['end_offset'])
                    chunks.append(data)
                except Exception as e:
                    log.warning(f"Byte-range failed {var_name}: {e}")
                break

    if not chunks:
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.grib2')
    for chunk in chunks:
        tmp.write(chunk)
    tmp.close()

    log.info(f"Got {len(chunks)}/{len(SURFACE_VARS)} vars: {member} f{fxx:03d}")
    return tmp.name
