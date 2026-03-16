import multiprocessing
import queue
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Any
import numpy as np

# Event data dtype (must match ExtractorWorker.EVENT_DTYPE)
EVENT_DTYPE = np.dtype([
    ("event_num", "<u8"),
    ("x", "<u2"),
    ("y", "<u2"),
    ("tof", "<f8"),
    ("tot", "<u4"),
])
# Pixel data dtype (no correlation, raw pixel data)
PIXEL_DTYPE = np.dtype([
    ("x", "<u2"),
    ("y", "<u2"),
    ("toa", "<f8"),
    ("tot", "<u4"),
])

class BaseSaverProcess(multiprocessing.Process, ABC):
    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        buffer_size: int = 8 * 1024 * 1024,
        log_level: str = "INFO",
        log_interval: int = 100,
        name: str = "BaseSaver",
        use_np_save: bool = False,  # Use np.save instead of raw struct I/O
    ):
        super().__init__(name=name, daemon=True)
        self.input_queue = input_queue
        self.buffer_size = buffer_size
        self.log_level = log_level
        self.log_interval = log_interval
        self.use_np_save = use_np_save
        self._stop_event = multiprocessing.Event()
        self._file = None
        self._buffer = None
        self._buffer_pos = 0
        self._total_items = 0
        self._flush_count = 0
        self._total_write_time = 0.0
        self._total_queue_time = 0.0
        self._start_time = None

    def stop(self):
        self._stop_event.set()

    @abstractmethod
    def _init_buffer(self):
        """Initialize the buffer for the specific data type."""
        pass

    @abstractmethod
    def _write_buffer_to_file(self, f, buffer, buffer_pos):
        """Write the buffer to the file (raw or np.save)."""
        pass

    @abstractmethod
    def _process_data_message(self, msg):
        """Process a data message (e.g., unpack and buffer)."""
        pass

    def _open_file(self, filepath: str):
        """Open a new file for writing."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        if self.use_np_save:
            self._file = open(filepath, "wb")
        else:
            self._file = open(filepath, "wb", buffering=self.buffer_size)
        self._total_items = 0
        self._flush_count = 0
        self._total_write_time = 0.0
        self._total_queue_time = 0.0
        self._start_time = time.time()
        self._logger.info(f"Recording to {filepath}")

    def _close_file(self):
        """Close the current file."""
        if self._file is not None:
            if self._buffer_pos > 0:
                self._write_buffer_to_file(self._file, self._buffer, self._buffer_pos)
            self._file.close()
            elapsed = time.time() - self._start_time
            rate_mb = (self._total_items * self._item_size / 1e6) / elapsed if elapsed > 0 else 0
            self._logger.info(f"Closed ({self._total_items:,} items at {rate_mb:.1f} MB/s)")
            self._file = None

    def run(self):
        from SERVAL.utils.logging import set_log_level, get_logger
        set_log_level(self.log_level)
        self._logger = get_logger(f"SERVAL.{self.name}")
        self._init_buffer()
        self._logger.info("Started (idle)")

        while not self._stop_event.is_set():
            try:
                t0 = time.perf_counter()
                msg = self.input_queue.get(timeout=0.1)
                t1 = time.perf_counter()
                self._total_queue_time += t1 - t0
            except queue.Empty:
                continue

            if msg == "STOP":
                break

            # Control messages
            if isinstance(msg, tuple) and msg and isinstance(msg[0], str):
                cmd = msg[0]
                if cmd == "NEW_FILE":
                    if self._file is not None:
                        self._close_file()
                    self._open_file(msg[1])
                elif cmd == "CLOSE_FILE":
                    if self._file is not None:
                        self._close_file()
                continue

            # Data message
            if self._file is None:
                continue

            try:
                self._process_data_message(msg)
            except Exception as e:
                self._logger.error(f"Error processing data: {e}", exc_info=True)

        # Cleanup
        if self._file is not None:
            self._close_file()
        self._logger.info("Stopped.")


class EventSaverProcess(BaseSaverProcess):
    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        buffer_size: int = 500_000,  # Number of events, not bytes
        log_level: str = "INFO",
        log_interval: int = 100,
        use_np_save: bool = False,
    ):
        super().__init__(
            input_queue=input_queue,
            buffer_size=buffer_size,
            log_level=log_level,
            log_interval=log_interval,
            name="EventSaverProcess",
            use_np_save=use_np_save,
        )
        self._item_size = 24  # Exact size of one event in bytes

    def _init_buffer(self):
        self._buffer = np.empty(self.buffer_size, dtype=EVENT_DTYPE)

    def _write_buffer_to_file(self, f, buffer, buffer_pos):
        if self.use_np_save:
            np.save(f, buffer[:buffer_pos])
        else:
            buffer[:buffer_pos].tofile(f)

    def _process_data_message(self, msg):
        event_num, x, y, tof, tot = msg
        n_events = len(event_num)
        self._total_items += n_events

        if self._buffer_pos + n_events > self.buffer_size:
            if self._buffer_pos > 0:
                self._write_buffer_to_file(self._file, self._buffer, self._buffer_pos)
                self._flush_count += 1
            self._buffer_pos = 0
        if n_events > self.buffer_size:
            records = np.empty(n_events, dtype=EVENT_DTYPE)
            records["event_num"] = event_num
            records["x"] = x
            records["y"] = y
            records["tof"] = tof
            records["tot"] = tot
            records.tofile(self._file)
            self._flush_count += 1
        else:
            end = self._buffer_pos + n_events
            self._buffer["event_num"][self._buffer_pos:end] = event_num
            self._buffer["x"][self._buffer_pos:end] = x
            self._buffer["y"][self._buffer_pos:end] = y
            self._buffer["tof"][self._buffer_pos:end] = tof
            self._buffer["tot"][self._buffer_pos:end] = tot
            self._buffer_pos = end


class RawSaverProcess(BaseSaverProcess):
    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        buffer_size: int = 8 * 1024 * 1024,
        log_level: str = "INFO",
        log_interval: int = 500,
        use_np_save: bool = False,
    ):
        super().__init__(
            input_queue=input_queue,
            buffer_size=buffer_size,
            log_level=log_level,
            log_interval=log_interval,
            name="RawSaverProcess",
            use_np_save=use_np_save,
        )
        self._item_size = 1  # Approximate size of one byte in bytes

    def _init_buffer(self):
        self._buffer = bytearray(self.buffer_size)

    def _write_buffer_to_file(self, f, buffer, buffer_pos):
        if self.use_np_save:
            np.save(f, np.frombuffer(buffer[:buffer_pos], dtype=np.uint8))
        else:
            f.write(buffer[:buffer_pos])

    def _process_data_message(self, msg):
        self._buffer[:len(msg)] = msg
        self._buffer_pos = len(msg)
        self._total_items += len(msg)
        self._write_buffer_to_file(self._file, self._buffer, self._buffer_pos)
        self._buffer_pos = 0


class PixelSaverProcess(BaseSaverProcess):
    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        buffer_size: int = 500_000,
        log_level: str = "INFO",
        log_interval: int = 100,
        use_np_save: bool = False,
    ):
        super().__init__(
            input_queue=input_queue,
            buffer_size=buffer_size,
            log_level=log_level,
            log_interval=log_interval,
            name="PixelSaverProcess",
            use_np_save=use_np_save,
        )
        self._item_size = 16  # Approximate size of one pixel in bytes

    def _init_buffer(self):
        self._buffer = np.empty(self.buffer_size, dtype=PIXEL_DTYPE)

    def _write_buffer_to_file(self, f, buffer, buffer_pos):
        if self.use_np_save:
            np.save(f, buffer[:buffer_pos])
        else:
            buffer[:buffer_pos].tofile(f)

    def _process_data_message(self, msg):
        x, y, toa, tot = msg
        n_pixels = len(x)
        self._total_items += n_pixels

        if self._buffer_pos + n_pixels > self.buffer_size:
            if self._buffer_pos > 0:
                self._write_buffer_to_file(self._file, self._buffer, self._buffer_pos)
                self._flush_count += 1
            self._buffer_pos = 0

        end = self._buffer_pos + n_pixels
        self._buffer["x"][self._buffer_pos:end] = x
        self._buffer["y"][self._buffer_pos:end] = y
        self._buffer["toa"][self._buffer_pos:end] = toa
        self._buffer["tot"][self._buffer_pos:end] = tot
        self._buffer_pos = end
