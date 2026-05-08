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

REFRESH_INTERVAL = 3600  # seconds — REFS runs every 6hr, refresh every 1hr

# -----------------------------------------------------------------------
# Background refresh
# -----------------------------------------------------------------------
def refresh_loop():
    while True:
        try:
            log.info("Starting REFS data refresh...")
            result = get_refs_probabilities()
            cache['threats']     = result['threats']
            cache['timeline']    = result['timeline']
            cache['cycle']       = result['cycle']
            cache['debug']       = result.get('debug', [])
            cache['last_updated'] = datetime.now(timezone.utc).isoformat()
            cache['error']       = None
            log.info(f"REFS refresh complete. Cycle: {result['cycle']}")
        except Exception as e:
            cache['error'] = str(e)
            log.error(f"REFS refresh failed: {e}")
        time.sleep(REFRESH_INTERVAL)

# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------
@app.route('/api/threats')
def threats():
    """Panel 1: max probability per hazard over 48-hour window."""
    if cache['threats']:
        return jsonify({
            'threats': cache['threats'],
            'cycle': cache['cycle'],
            'last_updated': cache['last_updated']
        })
    return jsonify({'error': cache['error'] or 'Data not yet available'}), 503


@app.route('/api/timeline')
def timeline():
    """Panel 2: hourly probability per hazard, hours 0-48."""
    if cache['timeline']:
        return jsonify({
            'timeline': cache['timeline'],
            'cycle': cache['cycle'],
            'last_updated': cache['last_updated']
        })
    return jsonify({'error': cache['error'] or 'Data not yet available'}), 503


@app.route('/api/status')
def status():
    """Health check and data freshness info."""
    return jsonify({
        'status': 'ok' if cache['threats'] else 'loading',
        'last_updated': cache['last_updated'],
        'cycle': cache['cycle'],
        'error': cache['error']
    })


@app.route('/api/debug')
def debug():
    """Diagnostic info — what files were found/tried."""
    return jsonify({
        'last_updated': cache['last_updated'],
        'cycle': cache['cycle'],
        'error': cache['error'],
        'debug_log': cache['debug']
    })


# -----------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------
if __name__ == '__main__':
    # Initial fetch before serving
    log.info("Performing initial REFS data fetch...")
    try:
        result = get_refs_probabilities()
        cache['threats']     = result['threats']
        cache['timeline']    = result['timeline']
        cache['cycle']       = result['cycle']
        cache['debug']       = result.get('debug', [])
        cache['last_updated'] = datetime.now(timezone.utc).isoformat()
        log.info(f"Initial fetch complete. Cycle: {result['cycle']}")
    except Exception as e:
        cache['error'] = str(e)
        log.error(f"Initial fetch failed: {e}")

    # Start background refresh thread
    t = threading.Thread(target=refresh_loop, daemon=True)
    t.start()

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
