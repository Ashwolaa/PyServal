#!/usr/bin/env python3
"""
Integration test / demo for SERVALController.

Requires a live SERVAL server at 192.168.1.1:8080.
Run with:  python scripts/test_serval_control.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qtpy.QtWidgets import QApplication
from SERVAL.controllers.serval_control import SERVALController
from SERVAL.utils.logging import set_log_level, get_logger

set_log_level('DEBUG')
logger = get_logger('test_serval_control')

_app = QApplication(sys.argv)
serval = SERVALController(host='192.168.1.1', port=8080)

logger.info("Testing SERVAL connection...")
if not serval.connect():
    logger.error("Connection failed (SERVAL may not be available)")
    sys.exit(1)

try:
    config = serval.get_config()
    logger.info("Got detector config: %s", list(config.keys()))
    config['GlobalTimestampInterval'] = 0.0
    serval.set_config(config)
except Exception as e:
    logger.error("Get config failed: %s", e)

serval.set_bias(50, enabled=True)
serval.set_trigger_settings('CONTINUOUS', -1, 0.5, 0.010)
serval.set_destination('192.168.1.2', 8088)

try:
    logger.debug("Destination: %s", json.dumps(serval.get_destination(), indent=2))
except Exception as e:
    logger.error("Get destination failed: %s", e)

try:
    logger.info("Dashboard keys: %s", list(serval.get_dashboard().keys()))
except Exception as e:
    logger.error("Get dashboard failed: %s", e)

logger.info("Starting measurement — Ctrl+C to stop")
serval.start_measurement()
try:
    while True:
        time.sleep(1)
        serval.get_dashboard()
except KeyboardInterrupt:
    pass

t0 = time.time()
serval.stop_measurement()
logger.info("Stopped in %.2f s", time.time() - t0)
