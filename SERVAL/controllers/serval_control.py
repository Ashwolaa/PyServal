#!/usr/bin/env python3
"""
SERVAL HTTP REST API Controller

Provides Qt-based controller for SERVAL detector via HTTP REST API.
SERVAL API base URL: http://192.168.1.1:8080

Features:
- Connection management
- Detector configuration (bias, triggers, DACs, BPC)
- Measurement control (start/stop)
- Data destination configuration
- Status monitoring
- Configurable logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)

Usage:
    from serval_control import SERVALController, set_log_level

    # Set logging level (default: INFO)
    set_log_level('DEBUG')  # or 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

    serval = SERVALController(host='192.168.1.1', port=8080)
    if serval.connect():
        serval.set_bias(100, enabled=True)
        serval.set_trigger_settings('AUTOTRIGSTART_TIMERSTOP', 20, 0.5, 0.010)
        serval.start_measurement()
"""

import json
from pathlib import Path

import requests
from qtpy.QtCore import QObject, Signal

from SERVAL.utils import get_logger, set_log_level

# Module logger
logger = get_logger('SERVAL.controller')

# Re-export for convenience
__all__ = ['SERVALController', 'set_log_level']


class SERVALController(QObject):
    """
    HTTP REST API client for SERVAL detector control

    SERVAL API Base: http://192.168.1.1:8080

    Signals:
        connected: Emitted when connection established
        disconnected: Emitted when connection lost
        config_changed: Emitted when configuration updated
        measurement_started: Emitted when measurement begins
        measurement_stopped: Emitted when measurement stops
        error_occurred: Emitted on API errors

    Examples:
        >>> serval = SERVALController()
        >>> serval.connected.connect(lambda: print("Connected!"))
        >>> serval.connect()
        >>> serval.set_bias(120)
        >>> serval.start_measurement()
    """

    # Qt signals
    connected = Signal()
    disconnected = Signal()
    config_changed = Signal(dict)
    measurement_started = Signal()
    measurement_stopped = Signal()
    error_occurred = Signal(str)

    def __init__(self, host='192.168.1.1', port=8080):
        """
        Initialize SERVAL controller

        Parameters:
            host (str): SERVAL server host/IP address
            port (int): SERVAL server port
        """
        super().__init__()
        self.host = host
        self.port = port
        self.base_url = f'http://{host}:{port}'
        self.is_connected = False
        self.timeout = 5

    def _get(self, endpoint, params=None, timeout=None):
        """
        Perform GET request to SERVAL API

        Parameters:
            endpoint (str): API endpoint (e.g., '/detector/config')
            params (dict): Optional query parameters
            timeout (int): Request timeout in seconds

        Returns:
            requests.Response: Response object

        Raises:
            requests.HTTPError: If request fails
        """
        response = requests.get(
            f'{self.base_url}{endpoint}',
            params=params,
            timeout=timeout or self.timeout
        )
        response.raise_for_status()
        return response

    def _put(self, endpoint, data=None, timeout=None):
        """
        Perform PUT request to SERVAL API

        Parameters:
            endpoint (str): API endpoint (e.g., '/detector/config')
            data (dict): JSON payload
            timeout (int): Request timeout in seconds

        Returns:
            requests.Response: Response object

        Raises:
            requests.HTTPError: If request fails
        """
        response = requests.put(
            f'{self.base_url}{endpoint}',
            json=data,
            timeout=timeout or self.timeout
        )
        response.raise_for_status()
        return response

    def connect(self):
        """
        Test connection to SERVAL server

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self._get('', timeout=2)
            self.is_connected = True
            self.connected.emit()
            logger.info("Connected to %s", self.base_url)
            return True
        except Exception as e:
            error_msg = f"Connection failed: {e}"
            self.error_occurred.emit(error_msg)
            logger.error(error_msg)
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from SERVAL server"""
        self.is_connected = False
        self.disconnected.emit()
        logger.info("Disconnected")

    def get_measurement_config(self):
        """
        Get current measurement configuration

        Returns:
            dict: Measurement configuration dictionary

        Raises:
            Exception: If request fails
        """
        try:
            return self._get('/measurement/config').json()
        except Exception as e:
            self.error_occurred.emit(f"Failed to get measurement config: {e}")
            raise

    def get_config(self):
        """
        Get current detector configuration

        Returns:
            dict: Configuration dictionary

        Raises:
            Exception: If request fails
        """
        try:
            return self._get('/detector/config').json()
        except Exception as e:
            self.error_occurred.emit(f"Failed to get config: {e}")
            raise

    def set_config(self, config_dict):
        """
        Update detector configuration

        Parameters:
            config_dict (dict): Configuration dictionary

        Returns:
            bool: True if successful
        """
        try:
            self._put('/detector/config', config_dict)
            self.config_changed.emit(config_dict)
            logger.info("Configuration updated")
            logger.debug("Config: %s", config_dict)
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to set config: {e}")
            logger.error("Config update failed: %s", e)
            return False

    def set_bias(self, voltage, enabled=True):
        """
        Set bias voltage

        Parameters:
            voltage (int): Bias voltage in volts (0-200)
            enabled (bool): Enable/disable bias

        Returns:
            bool: True if successful
        """
        try:
            config = self.get_config()
            config['BiasVoltage'] = voltage
            config['BiasEnabled'] = enabled
            success = self.set_config(config)

            if success:
                logger.info("Bias set to %dV (enabled=%s)", voltage, enabled)

            return success
        except Exception as e:
            self.error_occurred.emit(f"Failed to set bias: {e}")
            logger.error("Failed to set bias: %s", e)
            return False

    def set_trigger_settings(self, mode, n_triggers, period, exposure):
        """
        Configure trigger settings

        Parameters:
            mode (str): Trigger mode (e.g., 'AUTOTRIGSTART_TIMERSTOP', 'CONTINUOUS', 'EXTERNAL')
            n_triggers (int): Number of triggers
            period (float): Trigger period in seconds
            exposure (float): Exposure time in seconds

        Returns:
            bool: True if successful
        """
        try:
            config = self.get_config()
            config['TriggerMode'] = mode
            if mode != "CONTINUOUS":
                config['nTriggers'] = n_triggers
                config['TriggerPeriod'] = period
                config['ExposureTime'] = exposure
            success = self.set_config(config)

            if success:
                logger.info("Triggers: mode=%s, n=%d, period=%.3fs, exposure=%.3fs",
                           mode, n_triggers, period, exposure)

            return success
        except Exception as e:
            self.error_occurred.emit(f"Failed to set trigger settings: {e}")
            logger.error("Failed to set trigger settings: %s", e)
            return False

    def load_bpc(self, filepath):
        """
        Load pixel configuration file (.bpc)

        Parameters:
            filepath (str): Path to BPC file on SERVAL server

        Returns:
            bool: True if successful
        """
        try:
            self._get('/config/load', params={'format': 'pixelconfig', 'file': filepath}, timeout=10)
            logger.info("Loaded BPC file: %s", filepath)
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to load BPC: {e}")
            logger.error("BPC load failed: %s", e)
            return False

    def load_dacs(self, filepath):
        """
        Load DAC configuration file (.dacs)

        Parameters:
            filepath (str): Path to DACs file on SERVAL server

        Returns:
            bool: True if successful
        """
        try:
            self._get('/config/load', params={'format': 'dacs', 'file': filepath}, timeout=10)
            logger.info("Loaded DACs file: %s", filepath)
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to load DACs: {e}")
            logger.error("DACs load failed: %s", e)
            return False

    def set_destination(self, tcp_host=None, tcp_port=None, *,
                        destination=None, config_file=None, queue_size=16384):
        """
        Set data destination to TCP socket

        Configuration can be provided via:
        1. tcp_host/tcp_port parameters (creates default Raw TCP destination)
        2. destination dict (custom destination configuration)
        3. config_file path (JSON file with destination configuration)

        Parameters:
            tcp_host (str): Destination host IP
            tcp_port (int): Destination TCP port
            destination (dict): Custom destination configuration dict
            config_file (str|Path): Path to JSON config file with destination
            queue_size (int): Queue size for TCP destination (default: 16384)

        Returns:
            bool: True if successful

        Examples:
            # Using host/port
            serval.set_destination('192.168.1.2', 8088)

            # Using custom dict
            serval.set_destination(destination={'Raw': [...]})

            # Using config file
            serval.set_destination(config_file='/path/to/destination.json')
        """
        try:
            # Priority: config_file > destination dict > tcp_host/port
            if config_file is not None:
                config_path = Path(config_file)
                if not config_path.exists():
                    raise FileNotFoundError(f"Config file not found: {config_file}")
                with open(config_path) as f:
                    destination = json.load(f)
                logger.debug("Loaded destination config from: %s", config_file)

            elif destination is None:
                if tcp_host is None or tcp_port is None:
                    raise ValueError("Must provide tcp_host/tcp_port, destination dict, or config_file")
                destination = {
                    "Raw": [{
                        "Base": f"tcp://connect@{tcp_host}:{tcp_port}",
                        "FilePattern": "f%Hms_",
                        "QueueSize": queue_size
                    }]
                }

            self._put('/server/destination', destination)

            # Log what was set
            if tcp_host and tcp_port:
                logger.info("Data destination set to tcp://%s:%d", tcp_host, tcp_port)
            else:
                logger.info("Data destination configured")
            logger.debug("Destination config: %s", destination)
            return True

        except Exception as e:
            self.error_occurred.emit(f"Failed to set destination: {e}")
            logger.error("Set destination failed: %s", e)
            return False

    def get_destination(self):
        """
        Get current data destination configuration

        Returns:
            dict: Current destination configuration

        Raises:
            Exception: If request fails
        """
        try:
            return self._get('/server/destination').json()
        except Exception as e:
            self.error_occurred.emit(f"Failed to get destination: {e}")
            raise

    def start_measurement(self):
        """
        Start data acquisition on SERVAL

        Returns:
            bool: True if successful
        """
        try:
            self._get('/measurement/start')
            self.measurement_started.emit()
            logger.info("Measurement started")
            return True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 500:
                error_msg = (
                    "SERVAL returned 500 Server Error. "
                    "Possible causes: measurement already running, invalid configuration, or internal error"
                )
            else:
                error_msg = f"Failed to start measurement: {e}"
            self.error_occurred.emit(error_msg)
            logger.error("Start measurement failed: %s", error_msg)
            return False
        except Exception as e:
            self.error_occurred.emit(f"Failed to start measurement: {e}")
            logger.error("Start measurement failed: %s", e)
            return False

    def stop_measurement(self):
        """
        Stop data acquisition on SERVAL

        Returns:
            bool: True if successful
        """
        try:
            self._get('/measurement/stop')
            self.measurement_stopped.emit()
            logger.info("Measurement stopped")
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to stop measurement: {e}")
            logger.error("Stop measurement failed: %s", e)
            return False

    def get_dashboard(self):
        """
        Get current measurement status and dashboard info

        Returns:
            dict: Dashboard data with server info, detector info, measurement status

        Raises:
            Exception: If request fails
        """
        try:
            return self._get('/dashboard').json()
        except Exception as e:
            self.error_occurred.emit(f"Failed to get dashboard: {e}")
            raise


def test_serval_control():
    """Test SERVAL control module"""
    import sys
    import time
    from qtpy.QtWidgets import QApplication

    # Enable debug logging for tests
    set_log_level('DEBUG')

    _app = QApplication(sys.argv)  # Required for Qt signals

    serval = SERVALController(host='192.168.1.1', port=8080)

    logger.info("Testing SERVAL connection...")
    if not serval.connect():
        logger.error("Connection failed (SERVAL may not be available)")
        return

    # Test get config
    try:
        config = serval.get_config()
        logger.info("Got detector config: %s", list(config.keys()))
    except Exception as e:
        logger.error("Get config failed: %s", e)

    config['GlobalTimestampInterval'] = 0.0
    serval.set_config(config)
    # Test bias setting
    serval.set_bias(50, enabled=True)

    # Test measurement config
    try:
        config = serval.get_measurement_config()
        logger.info("Got measurement config: %s", list(config.keys()))
    except Exception as e:
        logger.error("Get measurement config failed: %s", e)

    # Test trigger settings
    serval.set_trigger_settings('CONTINUOUS', -1, 0.5, 0.010)

    # Test destination with host/port
    serval.set_destination('192.168.1.2', 8088)

    # Test destination with dict
    destination = {
        "Raw": [{
            "Base": "tcp://connect@192.168.1.2:8088",
            "FilePattern": "f%Hms_",
            "QueueSize": 16384,
        }],      
    }
    serval.set_destination(destination=destination)

    # Verify destination
    try:
        current_dest = serval.get_destination()
        logger.debug("Current destination: %s", json.dumps(current_dest, indent=2))
    except Exception as e:
        logger.error("Get destination failed: %s", e)

    # Test dashboard
    try:
        dashboard = serval.get_dashboard()
        logger.info("Dashboard keys: %s", list(dashboard.keys()))
    except Exception as e:
        logger.error("Get dashboard failed: %s", e)

    logger.info("All tests completed")

    # Run measurement loop
    measurement_duration = -1  # Duration in seconds (-1 = infinite, run until Ctrl+C)
    if measurement_duration == -1:
        logger.info("Running until Ctrl+C...")
    else:
        logger.info("Running for %d seconds...", measurement_duration)

    serval.start_measurement()
    try:
        start_time = time.time()
        while True:
            time.sleep(1)
            serval.get_dashboard()

            if measurement_duration > 0:
                elapsed = time.time() - start_time
                if elapsed >= measurement_duration:
                    logger.info("%d seconds elapsed - stopping measurement", measurement_duration)
                    break

                remaining = measurement_duration - elapsed
                if remaining > 0 and int(remaining) % 5 == 0:
                    logger.info("  %d seconds remaining...", int(remaining))
    except KeyboardInterrupt:
        logger.info("Ctrl+C detected")

    logger.info("Stopping measurement...")
    start = time.time()
    serval.stop_measurement()
    logger.info("Measurement stopped in %.2f seconds", time.time() - start)


if __name__ == '__main__':
    test_serval_control()
