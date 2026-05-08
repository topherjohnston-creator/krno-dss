"""
KRNO DSS Dashboard - Phase 2 Backend
Flask API that fetches REFS GRIB2 data from AWS S3,
computes hazard probabilities, and serves JSON to the dashboard.
"""

import os
import threading
import time
import logging
from datetime import datetime, timezone
from flask import Flask, jsonify
from flask_cors import CORS
from processor import get_refs_probabilities

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# --- App setup ---
app = Flask(__name__)
CORS(app)

# --- Cache ---
cache = {
    'threats': None,
    'timeline': None,
    'last_updated': None,
    'cycle': None,
    'error': None,
    'debug': []
}

REFRESH_INTERVAL = 3600  # 1 hour

# -----------------------------------------------------------------------
# Background refresh loop
# -----------------------------------------------------------------------
def refresh_loop():
    while True:
        try:
            log.info("Starting REFS data refresh...")
            result = get_refs_probabilities()
            cache['threats']      = result['threats']
            cache['timeline']     = result['timeline']
            cache['cycle']        = result['cycle']
            cache['debug']        = result.get('debug', [])
            cache['last_updated'] = datetime.now(timezone.utc).isoformat()
            cache['error']        = None
            log.info(f"REFS refresh complete. Cycle: {result['cycle']}")
        except Exception as e:
            cache['error'] = str(e)
            log.error(f"REFS refresh failed: {e}")
        time.sleep(REFRESH_INTERVAL)


# -----------------------------------------------------------------------
# Start background thread at MODULE LEVEL so gunicorn picks it up
# -----------------------------------------------------------------------
log.info("Starting background REFS fetch thread...")
_bg_thread = threading.Thread(target=refresh_loop, daemon=True)
_bg_thread.start()


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------
@app.route('/api/threats')
def threats():
    if cache['threats']:
        return jsonify({
            'threats': cache['threats'],
            'cycle': cache['cycle'],
            'last_updated': cache['last_updated']
        })
    return jsonify({'error': cache['error'] or 'Data not yet available — still loading'}), 503


@app.route('/api/timeline')
def timeline():
    if cache['timeline']:
        return jsonify({
            'timeline': cache['timeline'],
            'cycle': cache['cycle'],
            'last_updated': cache['last_updated']
        })
    return jsonify({'error': cache['error'] or 'Data not yet available — still loading'}), 503


@app.route('/api/status')
def status():
    return jsonify({
        'status': 'ok' if cache['threats'] else 'loading',
        'last_updated': cache['last_updated'],
        'cycle': cache['cycle'],
        'error': cache['error'],
        'thread_alive': _bg_thread.is_alive()
    })


@app.route('/api/debug')
def debug():
    return jsonify({
        'last_updated': cache['last_updated'],
        'cycle': cache['cycle'],
        'error': cache['error'],
        'debug_log': cache['debug'],
        'thread_alive': _bg_thread.is_alive()
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


@app.route('/api/files')
def files():
    """Show actual S3 files available for latest cycle member 1."""
    from downloader import s3_list, get_latest_cycle
    try:
        date_str, hour_str, base_path = get_latest_cycle()
        prefix = f"{base_path}m001/"
        keys = s3_list(prefix, max_keys=200)
        filenames = sorted([k.split('/')[-1] for k in keys if k.endswith('.grib2') and not k.endswith('.idx')])
        return jsonify({
            'cycle': f"{date_str} {hour_str}Z",
            'base_path': base_path,
            'prefix': prefix,
            'file_count': len(filenames),
            'files': filenames
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
