"""
Parameter schema for ServalAcquisitionGUI (ParameterManager).

Kept in a separate module so the ~175-line dict literal does not clutter
the main GUI class.  Import with ``from .acquisition_params import ACQUISITION_PARAMS``.
"""

from SERVAL.core.data_types import TDCChannel, TriggerEdge

ACQUISITION_PARAMS = [
    {'title': 'SERVAL', 'name': 'serval', 'type': 'group', 'children': [
        {'title': 'Connection', 'name': 'serval_connection', 'type': 'action_led', 'children': [
            {'title': 'Host', 'name': 'host', 'type': 'str', 'value': '192.168.1.1',
             'tip': 'IP address of the SERVAL server'},
            {'title': 'Port', 'name': 'port', 'type': 'int', 'value': 8080,
             'limits': (1, 65535), 'tip': 'SERVAL HTTP REST API port (default 8080)'},
        ]},
        {'title': 'Destination', 'name': 'serval_destination', 'type': 'action_led', 'children': [
            {'title': 'Dest. Host', 'name': 'dest_host', 'type': 'str', 'value': '192.168.1.2',
             'tip': 'IP address of this machine as seen from the SERVAL server'},
            {'title': 'Dest. Port', 'name': 'dest_port', 'type': 'int', 'value': 8088,
             'limits': (1, 65535), 'tip': 'TCP port on which the pipeline listens for incoming data'},
        ]},
        {'title': 'Voltage', 'name': 'voltage_settings', 'type': 'action_led', 'children': [
            {'title': 'Bias Voltage (V)', 'name': 'bias_voltage', 'type': 'int',
             'value': 40, 'limits': (0, 200),
             'tip': 'Sensor bias voltage. Typical range 40–100 V. Apply with the Voltage button.'},
            {'title': 'Bias Enabled', 'name': 'bias_enabled', 'type': 'bool', 'value': True,
             'tip': 'Enable or disable the bias voltage supply'},
        ]},
        {'title': 'Triggers', 'name': 'trigger_settings', 'type': 'action_led', 'children': [
            {'title': 'Trigger Mode', 'name': 'trigger_mode', 'type': 'list',
             'limits': ['CONTINUOUS', 'AUTOTRIGSTART_TIMERSTOP', 'EXTERNAL'
             'PEXSTART_NEXSTOP', 'NEXSTART_PEXSTOP', 'PEXSTART_TIMERSTOP', 'NEXSTART_TIMERSTOP',
             ],
             'value': 'CONTINUOUS',
             'tip': ('CONTINUOUS: free-running, no trigger correlation.\n'
                     'AUTOTRIGSTART_TIMERSTOP: internal timer generates triggers at Period/Exposure.\n'
                     'EXTERNAL: triggers come from the TDC input (hardware signal required).\n'
                     'PEXSTART_NEXSTOP: Acq. is started by positive edge external trigger input, stopped by negative edge.\n'
                     'NEXSTART_PEXSTOP: Acq. is started by negative edge external trigger input, stopped by positive edge.\n'
                     'PEXSTART_TIMERSTOP: Acq. is started by positive edge external trigger input, stopped by HW timer.\n'
                     'NEXSTART_TIMERSTOP: Acq. is started by negative edge external trigger input, stopped by HW timer.\n'
                     )},
            {'title': 'N Triggers', 'name': 'n_triggers', 'type': 'int', 'value': -1,
             'limits': (-1, 1000000), 'tip': 'Number of triggers before auto-stop. -1 = unlimited.'},
            {'title': 'Period (s)', 'name': 'trigger_period', 'type': 'float',
             'value': 0.5, 'limits': (0.001, 1000.0),
             'tip': 'Time between trigger starts (AUTOTRIGSTART_TIMERSTOP mode only)'},
            {'title': 'Exposure (s)', 'name': 'trigger_exposure', 'type': 'float',
             'value': 0.01, 'limits': (0.0001, 100.0),
             'tip': 'Active acquisition window per trigger (AUTOTRIGSTART_TIMERSTOP mode only)'},
        ]},
    ]},
    {'title': 'Acquisition', 'name': 'pipeline', 'type': 'group',
     'tip': 'Pipeline configuration — locked while acquisition is running', 'children': [
        {'title': 'Correlation', 'name': 'correlation', 'type': 'group', 'children': [
            {'title': 'TDC', 'name': 'tdc_id', 'type': 'list',
             'limits': TDCChannel.labels(), 'value': TDCChannel.TDC1.label,
             'tip': 'TDC channel carrying the trigger signal used for TOF correlation'},
            {'title': 'Edge', 'name': 'edge', 'type': 'list',
             'limits': TriggerEdge.labels(), 'value': TriggerEdge.RISING.label,
             'tip': 'Which edge of the TDC signal marks the trigger time (Rising or Falling)'},
            {'title': 'Event Window Min (ns)', 'name': 'event_window_min', 'type': 'float',
             'value': 0.0,
             'tip': 'Minimum TOF for a pixel to be correlated to a trigger (ns)'},
            {'title': 'Event Window Max (ns)', 'name': 'event_window_max', 'type': 'float',
             'value': 100000.0,
             'tip': 'Maximum TOF for a pixel to be correlated to a trigger (ns)'},
        ]},
        {'title': 'Extraction', 'name': 'processing', 'type': 'group', 'children': [
            {'title': 'Workers', 'name': 'num_workers', 'type': 'int', 'value': 4,
             'limits': (1, 16),
             'tip': 'Number of parallel extractor processes. Match to available CPU cores.'},
            {'title': 'Fast Extract', 'name': 'use_fast_extract', 'type': 'bool',
             'value': True,
             'tip': 'Use optimised (Numba JIT) extraction path. Disable only for debugging.'},
            {'title': 'Centroiding', 'name': 'centroiding', 'type': 'group', 'children': [
                {'title': 'Enable', 'name': 'use_centroiding', 'type': 'bool', 'value': False,
                 'tip': 'Cluster neighbouring pixel hits into single events (reduces charge sharing artefacts)'},
                {'title': 'Spatial eps (px)', 'name': 'eps_space', 'type': 'int',
                 'value': 2, 'limits': (1, 10),
                 'tip': 'Maximum pixel distance between hits in the same cluster'},
                {'title': 'Time eps (ns)', 'name': 'eps_time_ns', 'type': 'float',
                 'value': 100.0, 'limits': (1.0, 10000.0), 'step': 1.0, 'decimals': 0,
                 'tip': 'Maximum time difference between hits in the same cluster (ns)'},
            ]},
            {'title': 'Advanced', 'name': 'advanced', 'type': 'group', 'children': [
                {'title': 'ZMQ Port', 'name': 'zmq_port', 'type': 'int', 'value': 9200,
                 'limits': (1024, 65535),
                 'tip': 'Internal ZMQ port used between TCP receiver and extractor workers'},
                {'title': 'ZMQ HWM', 'name': 'zmq_hwm', 'type': 'int', 'value': 1000,
                 'limits': (10, 100000),
                 'tip': 'ZMQ high-water mark — max queued messages before chunks are dropped'},
                {'title': 'Triggers / Chunk', 'name': 'triggers_per_chunk', 'type': 'int',
                 'value': 100, 'limits': (0, 100_000),
                 'tip': ('Flush to workers every N rising edges of the selected TDC. '
                         'Each worker chunk is then guaranteed to contain exactly N '
                         'complete laser shots with no cross-chunk orphaned pixels. '
                         '0 = disabled, use chunk size / timeout only.')},
                {'title': 'Chunk Size (B)', 'name': 'chunk_size', 'type': 'int',
                 'value': 10_000_000, 'limits': (100_000, 100_000_000),
                 'tip': 'Fallback: flush after this many bytes when trigger-aligned flushing is disabled or not enough triggers have arrived yet'},
                {'title': 'Flush Timeout (s)', 'name': 'flush_timeout', 'type': 'float',
                 'value': 0.3, 'limits': (0.01, 5.0),
                 'tip': 'Force flush after this many seconds even if chunk size / trigger count not reached'},
                {'title': 'Recv Buffer (MB)', 'name': 'recv_buffer_mb', 'type': 'int',
                 'value': 2, 'limits': (1, 512),
                 'tip': 'OS-level TCP receive buffer size — increase if seeing dropped chunks at high rates'},
            ]},
        ]},
        {'title': 'Saving', 'name': 'saving', 'type': 'group', 'children': [
            {'title': 'Output Directory', 'name': 'output_dir', 'type': 'str',
             'value': './data',
             'tip': 'Root directory for saved data. Each recording creates a timestamped subdirectory here.'},
            {'title': 'Save Raw', 'name': 'save_raw', 'type': 'bool', 'value': True,
             'tip': 'Write raw TPX3 binary stream to .tpx3 file (needed to reprocess offline)'},
            {'title': 'Save Events', 'name': 'save_events', 'type': 'bool', 'value': True,
             'tip': 'Write correlated TOF events to _events.dat (numpy structured array)'},
            {'title': 'Save Pixels', 'name': 'save_pixels', 'type': 'bool', 'value': False,
             'tip': 'Write uncorrelated pixel hits to _pixels.dat (numpy structured array)'},
            {'title': 'Save Triggers', 'name': 'save_triggers', 'type': 'bool', 'value': True,
             'tip': 'Write TDC trigger timestamps to _triggers.dat'},
            {'title': 'Advanced', 'name': 'advanced', 'type': 'group', 'children': [
                {'title': 'Raw Savers', 'name': 'raw_num_savers', 'type': 'int',
                 'value': 1, 'limits': (0, 4),
                 'tip': 'Number of parallel processes writing raw data (0 disables raw saving)'},
                {'title': 'Events Savers', 'name': 'events_num_savers', 'type': 'int',
                 'value': 2, 'limits': (0, 8),
                 'tip': 'Number of parallel processes writing event data'},
                {'title': 'Pixels Savers', 'name': 'pixels_num_savers', 'type': 'int',
                 'value': 1, 'limits': (0, 4),
                 'tip': 'Number of parallel processes writing pixel data'},
                {'title': 'Triggers Savers', 'name': 'triggers_num_savers', 'type': 'int',
                 'value': 1, 'limits': (0, 4),
                 'tip': 'Number of parallel processes writing trigger data'},
            ]},
        ]},
        {'title': 'Live Feed', 'name': 'live', 'type': 'group', 'children': [
            {'title': 'Callback Mode', 'name': 'callback_mode', 'type': 'list',
             'limits': ['events', 'pixels', 'disabled'], 'value': 'events',
             'tip': 'What data the pipeline sends to the GUI for live display. Disable to reduce overhead.'},
        ]},
        {'title': 'External Control', 'name': 'external_control', 'type': 'group', 'children': [
            {'title': 'Command Server', 'name': 'command_server', 'type': 'group', 'children': [
                {'title': 'Enable', 'name': 'cmd_enabled', 'type': 'bool', 'value': True,
                 'tip': 'Enable ZMQ command server for external control (e.g. from PyMoDAQ)'},
                {'title': 'Port', 'name': 'cmd_port', 'type': 'int', 'value': 9100,
                 'limits': (1024, 65535),
                 'tip': 'ZMQ port for the command server'},
            ]},
        ]},
    ]},
    {'title': 'Display', 'name': 'display_settings', 'type': 'group', 'children': [
        {'title': 'Live Feed', 'name': 'live_feed', 'type': 'group', 'children': [
            {'title': 'Refresh Rate (s)', 'name': 'refresh_rate_s', 'type': 'float',
             'value': 1.0, 'limits': (0.1, 60.0),
             'tip': 'How often the histograms and images are redrawn (seconds)'},
            {'title': 'Live Subsampling (%)', 'name': 'display_fraction', 'type': 'float',
             'value': 100.0, 'limits': (1.0, 100.0), 'step': 5.0, 'decimals': 0,
             'tip': ('Percentage of incoming events/pixels fed to the live display. '
                     'Subsampling happens in the extractor workers before data is sent to the GUI, '
                     'so reducing this also lowers display lag at high rates. '
                     'Saving to disk is always full-resolution and unaffected.')},
            {'title': 'Auto-clear (s, -1=off)', 'name': 'clear_interval', 'type': 'float',
             'value': -1.0, 'limits': (-1.0, 3600.0),
             'tip': 'Automatically clear histogram every N seconds during acquisition. -1 disables.'},
            {'title': 'Time Window (s, -1=all)', 'name': 'max_time_window_s', 'type': 'float',
             'value': -1.0, 'limits': (-1.0, 3600.0),
             'tip': 'How many seconds of history the timeseries plots show. -1 = show all.'},
        ]},
        {'title': 'Appearance', 'name': 'appearance', 'type': 'group', 'children': [
            {'title': 'Colormap', 'name': 'colormap', 'type': 'list',
             'limits': ['viridis', 'plasma', 'inferno', 'magma', 'thermal'],
             'value': 'viridis', 'tip': 'Colour scale for the 2D pixel images'},
        ]},
        {'title': 'Mass Calibration', 'name': 'mass_calib', 'type': 'group', 'children': [
            {'title': 'Enable (show m/z)', 'name': 'enabled', 'type': 'bool', 'value': False,
             'tip': 'Display the histogram in calibrated mass units instead of TOF/TOA'},
            {'title': 'Coeff (ns / sqrt(mass))', 'name': 'coeff', 'type': 'float',
             'value': 1.0, 'tip': 'Calibration slope: tof_ns = coeff * sqrt(mass) + t0'},
            {'title': 't0 (ns)', 'name': 't0', 'type': 'float', 'value': 0.0,
             'tip': 'Calibration time offset: tof_ns = coeff * sqrt(mass) + t0'},
            {'title': 'Mass Min', 'name': 'mass_min', 'type': 'float', 'value': 0.0,
             'tip': 'Lower bound of the mass histogram axis'},
            {'title': 'Mass Max', 'name': 'mass_max', 'type': 'float', 'value': 200.0,
             'tip': 'Upper bound of the mass histogram axis'},
            {'title': 'Mass Bins', 'name': 'mass_bins', 'type': 'int', 'value': 1000,
             'limits': (100, 10000), 'tip': 'Number of bins in the mass histogram'},
        ]},
        {'title': 'Covariance Map', 'name': 'covariance', 'type': 'group', 'children': [
            {'title': 'Enable', 'name': 'cov_enabled', 'type': 'bool', 'value': False,
             'tip': ('Accumulate per-shot covariance map. '
                     'Requires trigger-correlated events (not pixel mode). '
                     'C(m1,m2) = <n(m1)·n(m2)> - <n(m1)>·<n(m2)>')},
        ]},
    ]},
]
