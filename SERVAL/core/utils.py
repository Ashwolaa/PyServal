from typing import Tuple, Optional, Dict, List
import numpy as np  


def find_last_pattern(view, pattern=b"TPX3") -> int:
    """
    Find the last occurrence of ``pattern`` in ``view``.

    Args:
        view: bytes or memoryview of the data
        pattern: byte sequence to search for

    Returns:
        int: Index of the last occurrence of ``pattern``, or -1 if not found.
    """
    if isinstance(view, memoryview):
        view = bytes(view)
    return view.rfind(pattern)