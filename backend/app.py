"""
KRNO DSS Dashboard - Phase 2 Backend
"""

import os
import threading
import time
import logging
from datetime import datetime, timezone
from flask import Flask, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

cache = {
    'threats': None, 'timeline': None, 'last_updated': None,
    'cycle': None, 'error': None, 'debug': []
}

REFRESH_INTERVAL = 3600


def do_refresh():
    """Single refresh attempt with granular logging."""
    live_log = ["Starting refresh..."]
    cache['debug'] = live_log
    try:
        from processor import get_refs_probabilities
        live_log.append("Processor imported OK")
        result = get_refs_probabilities(live_log=live_log)
        cache['threats']      = result['threats']
        cache['timeline']     = result['timeline']
        cache['cycle']        = result['cycle']
        cache['debug']        = live_log
        cache['last_updated'] = datetime.now(timezone.utc).isoformat()
        cache['error']        = None
        log.info(f"Refresh complete: {result['cycle']}")
    except Exception as e:
        import traceback
        err = f"{type(e).__name__}: {e}"
        tb  = traceback.format_exc()
        cache['error'] = err
        cache['debug'] = live_log + [f"ERROR: {err}", f"TRACEBACK: {tb}"]
        log.error(f"Refresh failed: {err}\n{tb}")


def refresh_loop():
    log.info("refresh_loop started")
    while True:
        log.info("Starting REFS refresh...")
        do_refresh()
        log.info(f"Sleeping {REFRESH_INTERVAL}s...")
        time.sleep(REFRESH_INTERVAL)


log.info("Starting background thread at module level...")
_bg_thread = threading.Thread(target=refresh_loop, daemon=True)
_bg_thread.start()


@app.route('/api/threats')
def threats():
    if cache['threats']:
        return jsonify({'threats': cache['threats'], 'cycle': cache['cycle'],
                        'last_updated': cache['last_updated']})
    return jsonify({'error': cache['error'] or 'Still loading'}), 503


@app.route('/api/timeline')
def timeline():
    if cache['timeline']:
        return jsonify({'timeline': cache['timeline'], 'cycle': cache['cycle'],
                        'last_updated': cache['last_updated']})
    return jsonify({'error': cache['error'] or 'Still loading'}), 503


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


@app.route('/api/files')
def files():
    from downloader import s3_list, get_latest_cycle
    try:
        date_str, hour_str, base_path = get_latest_cycle()
        prefix = f"{base_path}m001/"
        keys = s3_list(prefix, max_keys=200)
        filenames = sorted([k.split('/')[-1] for k in keys
                           if k.endswith('.grib2') and not k.endswith('.idx')])
        return jsonify({'cycle': f"{date_str} {hour_str}Z", 'base_path': base_path,
                        'file_count': len(filenames), 'files': filenames})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
